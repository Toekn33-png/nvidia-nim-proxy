
 // server.js - NVIDIA NIM proxy for Janitor AI
// Supports GLM 5.3, DeepSeek Flash, Kimi K2.6 and other NVIDIA NIM models.
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json({ limit: '25mb' }));

const PORT = process.env.PORT || 3000;
const API_KEY = process.env.NIM_API_KEY;
const API_BASE = (process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1').replace(/\/$/, '');
const TIMEOUT = Number(process.env.UPSTREAM_TIMEOUT_MS) || 90000;
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING = process.env.ENABLE_THINKING_MODE === 'true';

// Keep at 1 so Janitor AI uses the model you selected.
const MAX_ATTEMPTS = Math.max(1, Number(process.env.MAX_FALLBACK_ATTEMPTS) || 1);

const MODELS = {
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-pro': 'z-ai/glm-5.3',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'deepseek-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4.1-flash',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'kimi-k3': 'moonshotai/kimi-k3',
  'kimi': 'moonshotai/kimi-k3',
  'nemotron-ultra': 'nvidia/nemotron-3-ultra-550b-a55b',
  'nemotron-super': 'nvidia/nemotron-3-super-120b-a12b',
  'nemotron-super-vl': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'nemotron-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'step-5': 'stepfun-ai/step-5-preview',
  'step-flash': 'stepfun-ai/step-5-preview',
  'gemma': 'google/gemma-4-31b-it',
  'muse-glimmer': 'meta/muse-glimmer-30b',
  'laguna': 'poolside/laguna-xs-2.1',
  'gpt-oss': 'openai/gpt-oss-20b',
  'mistral-medium': 'mistralai/mistral-large',
  'mistral-small': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-4': 'z-ai/glm-5.3',
  'gpt-4o': 'z-ai/glm-5.3',
  'gpt-4-faster': 'z-ai/glm-5.3-flash',
  'gpt-4o-mini': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gpt-3.5-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus': 'moonshotai/kimi-k3',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'z-ai/glm-5.3',
  'minimax': 'z-ai/glm-5.3',
  'inkling': 'z-ai/glm-5.3'
};

function resolveModel(raw) {
  let name = String(raw || 'glm-5.3').trim().toLowerCase();

  if (MODELS[name]) return MODELS[name];

  // Fix bad names such as "nvidia/deepseek-flash".
  if (name.startsWith('nvidia/') && MODELS[name.slice(7)]) {
    return MODELS[name.slice(7)];
  }

  // Allow actual NVIDIA catalog IDs.
  if (name.includes('/')) return name;

  const error = new Error(
    `Unknown model: ${name}. Try glm-5.3, deepseek-flash, or kimi-k2.6.`
  );
  error.status = 400;
  throw error;
}

const ROLEPLAY_RULES = `Roleplay only as the character established by the conversation and character card.
Keep the character's personality, beliefs, flaws, speaking style and memories consistent.
Use believable, natural conversation instead of flowery narration or dramatic speeches.
Do not assume what the user thinks, feels, says or does. Never write the user's turn.
Let relationships and emotions evolve naturally, without rushing them.
End after the character's turn and wait for the user's response.`;

function cleanText(text) {
  return String(text || '')
    .split(/\n\s*(?:User|Human|Player)\s*[:：]/i)[0]
    .trimEnd();
}

function addRoleplayRules(messages) {
  const out = messages.map(m => ({ ...m }));
  const i = out.findIndex(m => m.role === 'system');

  if (i >= 0 && typeof out[i].content === 'string') {
    out[i].content += '\n\n' + ROLEPLAY_RULES;
  } else {
    out.unshift({ role: 'system', content: ROLEPLAY_RULES });
  }

  return out;
}

function enableThinking(request) {
  if (!ENABLE_THINKING) return;

  const id = request.model;

  if (
    id.startsWith('deepseek-ai/') ||
    id.startsWith('moonshotai/')
  ) {
    request.chat_template_kwargs = { thinking: true };
  } else if (
    id.startsWith('z-ai/') ||
    id.startsWith('nvidia/') ||
    id.startsWith('stepfun-ai/')
  ) {
    request.chat_template_kwargs = { enable_thinking: true };
  }
}

const FALLBACKS = [
  'z-ai/glm-5.3-flash',
  'deepseek-ai/deepseek-v4.1-flash',
  'nvidia/nemotron-3.5-lightning-30b-a3b'
];

async function requestNim(body) {
  const models = [
    body.model,
    ...FALLBACKS.filter(id => id !== body.model)
  ].slice(0, MAX_ATTEMPTS);

  let lastError;

  for (const model of models) {
    const payload = { ...body, model };
    enableThinking(payload);

    try {
      console.log('Requesting NVIDIA model:', model);

      const response = await axios.post(
        `${API_BASE}/chat/completions`,
        payload,
        {
          headers: {
            Authorization: `Bearer ${API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: body.stream ? 'stream' : 'json',
          timeout: TIMEOUT
        }
      );

      return { response, usedModel: model };

    } catch (err) {
      lastError = err;

      if (
        ![404, 410, 429, 500, 502, 503, 504].includes(
          err.response?.status
        ) && err.code !== 'ECONNABORTED'
      ) {
        break;
      }
    }
  }

  throw lastError;
}

app.get('/', (_req, res) => res.json({
  service: 'Janitor AI NVIDIA NIM Proxy',
  version: '3.1-multi-model',
  models: {
    'glm-5.3': MODELS['glm-5.3'],
    'deepseek-flash': MODELS['deepseek-flash'],
    'kimi-k2.6': MODELS['kimi-k2.6']
  },
  endpoint: '/v1/chat/completions'
}));

app.get('/health', (_req, res) => res.json({
  status: 'ok',
  version: '3.1-multi-model',
  api_key_configured: !!API_KEY
}));

app.get('/resolve-model', (req, res) => {
  try {
    res.json({
      requested: req.query.model || 'glm-5.3',
      resolved: resolveModel(req.query.model)
    });
  } catch (err) {
    res.status(err.status || 400).json({
      error: err.message
    });
  }
});

app.get('/v1/models', (_req, res) => res.json({
  object: 'list',
  data: Object.keys(MODELS).map(id => ({
    id,
    object: 'model',
    owned_by: 'nvidia-nim-proxy'
  }))
}));

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!API_KEY) {
      return res.status(500).json({
        error: {
          message: 'Set NIM_API_KEY in Render Environment.'
        }
      });
    }

    const {
      model,
      messages,
      stream,
      temperature,
      max_tokens,
      top_p,
      frequency_penalty,
      presence_penalty,
      stop,
      seed
    } = req.body || {};

    if (!Array.isArray(messages) || !messages.length) {
      return res.status(400).json({
        error: {
          message: 'messages must be a nonempty array.'
        }
      });
    }

    const payload = {
      model: resolveModel(model),
      messages: addRoleplayRules(messages),
      stream: stream === true,
      temperature: temperature ?? 0.75,
      max_tokens: max_tokens ?? 1200
    };

    for (const [key, value] of Object.entries({
      top_p,
      frequency_penalty,
      presence_penalty,
      stop,
      seed
    })) {
      if (value !== undefined) {
        payload[key] = value;
      }
    }

    const { response, usedModel } = await requestNim(payload);

    if (!payload.stream) {
      const data = response.data;

      if (Array.isArray(data.choices)) {
        data.choices = data.choices.map(choice => ({
          ...choice,
          message: {
            ...choice.message,
            content: cleanText(choice.message?.content),
            ...(SHOW_REASONING ? {} : {
              reasoning: undefined,
              reasoning_content: undefined
            })
          }
        }));
      }

      data.model = usedModel;
      return res.json(data);
    }

    // OpenAI-compatible streaming responses.
    res.setHeader(
      'Content-Type',
      'text/event-stream; charset=utf-8'
    );
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    let buffer = '';
    let pending = '';
    let finished = false;
    let stopped = false;

    const send = content => {
      if (!content || res.writableEnded) return;

      res.write(`data: ${JSON.stringify({
        object: 'chat.completion.chunk',
        model: usedModel,
        choices: [{
          index: 0,
          delta: { content },
          finish_reason: null
        }]
      })}\n\n`);
    };

    const flush = final => {
      if (stopped && !final) return;

      const match =
        /(?:^|\n)\s*(?:User|Human|Player)\s*[:：]/i.exec(
          pending
        );

      if (match) {
        pending = pending.slice(0, match.index);
        stopped = true;
        final = true;
      }

      const count = final
        ? pending.length
        : Math.max(0, pending.length - 180);

      if (count) {
        send(pending.slice(0, count));
        pending = pending.slice(count);
      }
    };

    const finish = () => {
      if (finished) return;

      finished = true;
      flush(true);

      res.write(`data: ${JSON.stringify({
        object: 'chat.completion.chunk',
        model: usedModel,
        choices: [{
          index: 0,
          delta: {},
          finish_reason: 'stop'
        }]
      })}\n\n`);

      res.end('data: [DONE]\n\n');
    };

    response.data.on('data', chunk => {
      buffer += chunk.toString();

      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;

        const raw = line.slice(5).trim();

        if (raw === '[DONE]') {
          finish();
          return;
        }

        try {
          const parsed = JSON.parse(raw);
          const delta = parsed.choices?.[0]?.delta || {};

          if (stopped) continue;

          if (SHOW_REASONING) {
            pending +=
              delta.reasoning_content ||
              delta.reasoning ||
              '';
          }

          if (typeof delta.content === 'string') {
            pending += delta.content;
          }

          flush(false);

        } catch {
          // Ignore malformed streaming lines.
        }
      }
    });

    response.data.on('end', finish);

    response.data.on('error', err => {
      console.error(
        'Upstream streaming error:',
        err.message
      );
      finish();
    });

  } catch (err) {
    const status =
      err.status ||
      err.response?.status ||
      500;

    console.error(
      'NVIDIA proxy error:',
      status,
      err.message
    );

    if (!res.headersSent) {
      res.status(status).json({
        error: {
          message:
            err.response?.data?.detail ||
            err.message,
          code: status
        }
      });
    } else {
      res.end();
    }
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(
    `Janitor AI proxy 3.1-multi-model listening on ${PORT}`
  );
  console.log(
    'Models: glm-5.3, deepseek-flash, kimi-k2.6'
  );
});
