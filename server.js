
 // NVIDIA NIM -> Janitor AI (OpenAI-compatible) proxy
// Replace your old server.js with this file.
// Requires: npm install express cors axios

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const PORT = Number(process.env.PORT) || 3000;
const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_API_BASE = (process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1').replace(/\/+$/, '');
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const GLM_REASONING_EFFORT = ENABLE_THINKING_MODE ? 'high' : 'low';

// Short defaults for troubleshooting. Raise limits after it works consistently.
const MAX_FALLBACK_ATTEMPTS = Math.max(1, Math.min(3, Number(process.env.MAX_FALLBACK_ATTEMPTS) || 1));
const UPSTREAM_TIMEOUT_MS = Math.max(10000, Number(process.env.UPSTREAM_TIMEOUT_MS) || 40000);
const MAX_REQUEST_MS = Math.max(UPSTREAM_TIMEOUT_MS, Number(process.env.MAX_REQUEST_MS) || 75000);
const MAX_OUTPUT_TOKENS = Math.max(512, Number(process.env.MAX_OUTPUT_TOKENS) || 4096);

// These are ALIASES, not actual Claude/OpenAI API connections.
const MODEL_MAPPING = {
  'claude-sonnet': 'z-ai/glm-5.3',
  'claude-3-sonnet': 'z-ai/glm-5.3',
  'claude-3-opus': 'z-ai/glm-5.3',
  'gpt-4': 'z-ai/glm-5.3',
  'gpt-4o': 'z-ai/glm-5.3',
  'gemini-pro': 'z-ai/glm-5.3',
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-pro': 'z-ai/glm-5.3',
  'glm-flash': 'z-ai/glm-5.3-flash',
  'gpt-4-faster': 'z-ai/glm-5.3-flash',
  'gpt-4o-mini': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nemotron-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nemotron-super': 'nvidia/nemotron-3-super-120b-a12b',
  'nemotron-ultra': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'deepseek-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4.1-flash',
  'kimi': 'moonshotai/kimi-k2.6',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'kimi-k3': 'moonshotai/kimi-k3',
  'step-5': 'stepfun-ai/step-5-preview',
  'step-flash': 'stepfun-ai/step-5-preview',
  'gpt-oss': 'openai/gpt-oss-20b',
  'minimax': 'z-ai/glm-5.3',
  'mistral-medium': 'mistralai/mistral-large',
  'mistral-small': 'nvidia/nemotron-3.5-lightning-30b-a3b'
};

const FALLBACK_CHAIN = [
  'z-ai/glm-5.3-flash',
  'nvidia/nemotron-3.5-lightning-30b-a3b'
];

const FALLBACK_STATUSES = new Set([404, 410, 429, 500, 502, 503, 504]);

const RP_RULES = [
  'Write only for the character(s) controlled by the assistant.',
  'Do not write the user character\'s dialogue, actions, decisions, or thoughts.',
  'Keep dialogue natural and character-consistent. Avoid repetitive descriptions.',
  'Use meaningful detail and maintain continuity from earlier messages.'
].join(' ');

function resolveModel(name) {
  if (!name || typeof name !== 'string') return 'z-ai/glm-5.3-flash';
  const key = name.trim().toLowerCase();
  if (MODEL_MAPPING[key]) return MODEL_MAPPING[key];
  if (key.includes('/')) return name.trim();
  if (key.includes('claude') || key.includes('gpt-4')) return 'z-ai/glm-5.3';
  console.warn(`Unknown model alias '${name}', using GLM Flash`);
  return 'z-ai/glm-5.3-flash';
}

function addRoleplayRules(messages) {
  const copy = messages.map(m => ({ ...m }));
  const system = copy.find(m => m.role === 'system' && typeof m.content === 'string');
  if (system) system.content += '\n\n' + RP_RULES;
  else copy.unshift({ role: 'system', content: RP_RULES });
  return copy;
}

function makeRequest(body, model) {
  const clientMax = Number(body.max_tokens);
  const maxTokens = Number.isFinite(clientMax) && clientMax > 0
    ? Math.min(Math.floor(clientMax), MAX_OUTPUT_TOKENS)
    : Math.min(3072, MAX_OUTPUT_TOKENS);

  const temp = Number(body.temperature);

  const request = {
    model,
    messages: addRoleplayRules(body.messages),
    stream: body.stream === true,
    temperature: body.temperature == null || !Number.isFinite(temp)
      ? 0.7
      : Math.min(1, Math.max(0, temp)),
    max_tokens: maxTokens
  };

  // GLM-5.3 uses lighter reasoning when Thinking Mode is disabled.
  if (model === 'z-ai/glm-5.3' || model === 'z-ai/glm-5.3-flash') {
    request.chat_template_kwargs = {
      reasoning_effort: GLM_REASONING_EFFORT,
      clear_thinking: true
    };
  }

  // Pass through optional Janitor AI settings.
  for (const key of [
    'top_p',
    'frequency_penalty',
    'presence_penalty',
    'stop',
    'seed'
  ]) {
    if (body[key] !== undefined && body[key] !== null) {
      request[key] = body[key];
    }
  }

  return request;
}

function errorDetails(err) {
  const status = err.response?.status;

  if (status === 401) {
    return {
      status: 401,
      message: 'Invalid NVIDIA API key. Check NIM_API_KEY in Render.'
    };
  }

  if (status === 402) {
    return {
      status: 402,
      message: 'NVIDIA API credits are unavailable.'
    };
  }

  if (status === 404 || status === 410) {
    return {
      status: 404,
      message: 'NVIDIA does not offer this model right now. Try glm-flash.'
    };
  }

  if (status === 429) {
    return {
      status: 429,
      message: 'NVIDIA rate limit reached. Wait and retry.'
    };
  }

  if (status === 400 || status === 422) {
    return {
      status,
      message: `NVIDIA rejected the request (HTTP ${status}). Try glm-flash or check model settings.`
    };
  }

  if (status === 503) {
    return {
      status: 503,
      message: 'NVIDIA model is currently overloaded. Try glm-flash.'
    };
  }

  if (
    err.code === 'ECONNABORTED' ||
    err.code === 'ETIMEDOUT' ||
    err.code === 'ERR_CANCELED'
  ) {
    return {
      status: 504,
      message: 'NVIDIA took too long to reply. Try glm-flash, a shorter chat context, or retry later.'
    };
  }

  return {
    status: status || 502,
    message: `Proxy/NVIDIA error: ${err.message}`
  };
}

function writeEvent(res, value) {
  if (!res.destroyed && !res.writableEnded) {
    res.write(`data: ${JSON.stringify(value)}\n\n`);
  }
}

function writeText(res, text, model) {
  writeEvent(res, {
    id: `chatcmpl-proxy-${Date.now()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta: { content: text },
        finish_reason: null
      }
    ]
  });
}

function finishStream(res) {
  if (!res.destroyed && !res.writableEnded) {
    res.write('data: [DONE]\n\n');
    res.end();
  }
}

async function relayStream(upstream, res, model) {
  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');

  res.flushHeaders();
  res.write(': connected\n\n');

  // Heartbeats help keep the connection alive while NVIDIA thinks.
  const heartbeat = setInterval(() => {
    if (!res.destroyed && !res.writableEnded) {
      res.write(': keep-alive\n\n');
    }
  }, 10000);

  const decoder = new StringDecoder('utf8');

  let buffer = '';
  let receivedDone = false;
  let deliveredText = false;
  let reasoningOpen = false;

  function processEvent(raw) {
    const payloads = raw
      .split('\n')
      .filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).trimStart());

    for (const payload of payloads) {
      if (payload === '[DONE]') {
        receivedDone = true;

        if (SHOW_REASONING && reasoningOpen) {
          writeText(res, '\n</think>', model);
          reasoningOpen = false;
        }

        if (!deliveredText) {
          writeText(
            res,
            '[The model returned no visible reply. Try glm-flash or increase the token limit.]',
            model
          );
        }

        finishStream(res);
        return;
      }

      let data;

      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }

      if (!Array.isArray(data.choices)) continue;

      let shouldForward = false;

      for (const choice of data.choices) {
        const delta = choice.delta;
        if (!delta) continue;

        const reasoning = delta.reasoning_content ?? delta.reasoning;
        const visible =
          typeof delta.content === 'string' ? delta.content : '';

        let outgoing = visible;

        if (SHOW_REASONING) {
          if (reasoning) {
            if (!reasoningOpen) {
              outgoing = '<think>\n' + reasoning + outgoing;
              reasoningOpen = true;
            } else {
              outgoing = reasoning + outgoing;
            }
          }

          if (visible && reasoningOpen) {
            outgoing =
              (reasoning
                ? outgoing.slice(0, outgoing.length - visible.length)
                : '') +
              '\n</think>\n\n' +
              visible;

            reasoningOpen = false;
          }
        }

        delete delta.reasoning_content;
        delete delta.reasoning;

        delta.content = outgoing;

        if (outgoing) {
          shouldForward = true;
          deliveredText = true;
        }

        if (delta.role || choice.finish_reason != null) {
          shouldForward = true;
        }
      }

      if (shouldForward) {
        writeEvent(res, data);
      }
    }
  }

  try {
    for await (const chunk of upstream.data) {
      if (res.destroyed || res.writableEnded) break;

      buffer = (buffer + decoder.write(chunk)).replace(/\r\n/g, '\n');

      let index;

      while ((index = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);

        processEvent(event);

        if (receivedDone || res.destroyed) break;
      }

      if (receivedDone || res.destroyed) break;
    }

    if (!receivedDone && !res.destroyed) {
      buffer += decoder.end();

      if (buffer.trim()) {
        processEvent(buffer.replace(/\r\n/g, '\n'));
      }

      if (!receivedDone && !res.writableEnded) {
        writeText(
          res,
          deliveredText
            ? '\n[Response stream ended unexpectedly.]'
            : '[NVIDIA returned an incomplete reply. Try glm-flash.]',
          model
        );

        finishStream(res);
      }
    }
  } finally {
    clearInterval(heartbeat);
    upstream.data.destroy();
  }
}

// Health and model endpoints.
app.get('/', (req, res) => {
  res.json({
    status: 'running',
    service: 'NVIDIA NIM / Janitor AI proxy',
    chat: '/v1/chat/completions'
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    nim_api_configured: Boolean(NIM_API_KEY),
    thinking_mode: ENABLE_THINKING_MODE,
    glm_reasoning_effort: GLM_REASONING_EFFORT,
    show_reasoning: SHOW_REASONING,
    max_fallback_attempts: MAX_FALLBACK_ATTEMPTS,
    upstream_timeout_ms: UPSTREAM_TIMEOUT_MS
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.entries(MODEL_MAPPING).map(([id, nim_model]) => ({
      id,
      nim_model,
      object: 'model',
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

// Main Janitor AI chat endpoint.
app.post('/v1/chat/completions', async (req, res) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({
      error: {
        message: 'Missing NIM_API_KEY in Render.'
      }
    });
  }

  if (
    !Array.isArray(req.body?.messages) ||
    req.body.messages.length === 0
  ) {
    return res.status(400).json({
      error: {
        message: 'messages must be a nonempty array.'
      }
    });
  }

  const isStreaming = req.body.stream === true;
  const selected = resolveModel(req.body.model);

  const candidates = [
    selected,
    ...FALLBACK_CHAIN.filter(m => m !== selected)
  ].slice(0, MAX_FALLBACK_ATTEMPTS);

  const controller = new AbortController();

  const totalTimeout = setTimeout(() => {
    controller.abort();
  }, MAX_REQUEST_MS);

  // Abort when the client disconnects.
  res.on('close', () => {
    if (!res.writableEnded) {
      controller.abort();
    }
  });

  let modelUsed = selected;

  try {
    let upstream;

    for (let i = 0; i < candidates.length; i++) {
      modelUsed = candidates[i];

      const payload = makeRequest(req.body, modelUsed);

      console.log(
        `Trying ${modelUsed} (attempt ${i + 1}/${candidates.length}, stream=${isStreaming}, max_tokens=${payload.max_tokens})`
      );

      try {
        upstream = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          payload,
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json',
              Accept: isStreaming
                ? 'text/event-stream'
                : 'application/json'
            },
            responseType: isStreaming ? 'stream' : 'json',
            timeout: UPSTREAM_TIMEOUT_MS,
            signal: controller.signal
          }
        );

        break;
      } catch (err) {
        console.error(
          `${modelUsed} failed: HTTP ${err.response?.status || '-'}, ${err.message}`
        );

        err.response?.data?.destroy?.();

        const retryable =
          FALLBACK_STATUSES.has(err.response?.status) ||
          ['ECONNABORTED', 'ETIMEDOUT'].includes(err.code);

        if (
          !retryable ||
          i === candidates.length - 1 ||
          controller.signal.aborted
        ) {
          throw err;
        }
      }
    }

    if (isStreaming) {
      await relayStream(upstream, res, modelUsed);
    } else {
      const data = upstream.data;

      if (!Array.isArray(data?.choices)) {
        throw new Error('Unexpected NVIDIA response format');
      }

      for (const choice of data.choices) {
        if (choice.message) {
          const reasoning =
            choice.message.reasoning_content ??
            choice.message.reasoning;

          choice.message.content =
            typeof choice.message.content === 'string'
              ? choice.message.content
              : '';

          if (SHOW_REASONING && reasoning) {
            choice.message.content =
              `<think>\n${reasoning}\n</think>\n\n` +
              choice.message.content;
          }

          delete choice.message.reasoning_content;
          delete choice.message.reasoning;

          if (!choice.message.content) {
            choice.message.content =
              '[The model returned no visible reply. Try glm-flash.]';
          }
        }
      }

      res.json({
        ...data,
        model: req.body.model || modelUsed
      });
    }
  } catch (err) {
    if (res.destroyed) return;

    const detail = errorDetails(err);

    console.error(`Proxy error: ${detail.message}`);

    if (res.headersSent) {
      writeText(res, `\n[${detail.message}]`, modelUsed);
      finishStream(res);
    } else {
      if (detail.status === 429) {
        res.setHeader('Retry-After', '60');
      }

      res.status(detail.status).json({
        error: {
          message: detail.message,
          type: 'api_error',
          code: detail.status
        }
      });
    }
  } finally {
    clearTimeout(totalTimeout);
  }
});

// Compatible with Express 4 and Express 5.
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Unknown endpoint ${req.path}`
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`NVIDIA Janitor AI proxy listening on port ${PORT}`);
  console.log(`NIM API key configured: ${Boolean(NIM_API_KEY)}`);
  console.log(
    `GLM reasoning: ${GLM_REASONING_EFFORT}; reasoning display: ${SHOW_REASONING}`
  );
  console.log(
    `Fallback attempts: ${MAX_FALLBACK_ATTEMPTS}; per-call timeout: ${UPSTREAM_TIMEOUT_MS}ms`
  );
});
