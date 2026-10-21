// server.js - OpenAI to NVIDIA NIM API Proxy (Optimized for Janitor AI)
// Updated: October 16, 2026 — model IDs verified against build.nvidia.com / integrate.api.nvidia.com/v1/models
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = process.env.SHOW_REASONING === 'true' || false;

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true' || false;

// How many models (primary + fallbacks) to try per request. NVIDIA's free tier is ~40 RPM,
// so don't hammer it with a long chain on every 429.
const MAX_FALLBACK_ATTEMPTS = parseInt(process.env.MAX_FALLBACK_ATTEMPTS, 10) || 4;

// Upstream request timeout (ms). A hung model falls through to the next one.
const UPSTREAM_TIMEOUT_MS = parseInt(process.env.UPSTREAM_TIMEOUT_MS, 10) || 60000;

// If true, still try mapped models that are missing from NVIDIA's public model list (e.g. gated ones like Kimi K3)
const ALLOW_UNLISTED_MODELS = process.env.ALLOW_UNLISTED_MODELS === 'true';

// 🎯 MODEL MAPPING — updated October 2026
//
// REMOVED from NIM since the last version of this file:
//   • minimaxai/minimax-m3           — gone from the catalog
//   • openai/gpt-oss-120b            — gone (only gpt-oss-20b remains)
//   • deepseek-ai/deepseek-v4-pro    — deprecated Aug 7 2026, replaced by deepseek-v4-pro-0813
//   • z-ai/glm-5.2, stepfun step-3.7-flash, mistral-medium-3.5 / small-4, inkling — no longer in the
//     current model list; superseded by GLM-5.3, Step-5 preview, etc.
// The old Janitor alias names (minimax, glm-pro, step-flash, ...) are kept and redirected to the
// closest live replacement so existing Janitor AI configs don't break.
const MODEL_MAPPING = {
  // --- DeepSeek ---
  // NOTE: deepseek-v4-pro-0813 returned 410 (Gone) on the API and is not in the public list, so Pro aliases use V4.1 Flash
  'deepseek-v4-pro':    'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-flash':     'deepseek-ai/deepseek-v4.1-flash',    // 552B MoE / 8B active, 1M ctx, multimodal
  'gpt-4':              'z-ai/glm-5.3',

  // --- Moonshot Kimi ---
  'kimi-k3':            'moonshotai/kimi-k3',                 // ~2.8T MoE, 1M ctx (heavily rate limited on free tier)
  'kimi':               'moonshotai/kimi-k3',
  'kimi-k2.6':          'moonshotai/kimi-k2.6',               // 262K ctx, more reliable than K3
  'claude-3-opus':      'moonshotai/kimi-k3',

  // --- GLM (Z.ai) ---
  'glm-5.3':            'z-ai/glm-5.3',                       // 1M ctx, #2 open model on AA index
  'glm-pro':            'z-ai/glm-5.3',                       // legacy alias (was glm-5.2)
  'glm-flash':          'z-ai/glm-5.3-flash',                 // 1M ctx, fast
  'gpt-4o':             'z-ai/glm-5.3',                       // (was minimax-m3)
  'gpt-4-faster':       'z-ai/glm-5.3-flash',                 // (was step-3.7-flash)
  'minimax':            'z-ai/glm-5.3',                       // legacy alias — MiniMax removed from NIM

  // --- NVIDIA Nemotron ---
  'gpt-3.5-turbo':      'nvidia/nemotron-3-ultra-550b-a55b',  // 550B MoE, 1M ctx
  'nemotron-ultra':     'nvidia/nemotron-3-ultra-550b-a55b',
  'nemotron-super':     'nvidia/nemotron-3-super-120b-a12b',  // 262K ctx
  'nemotron-super-vl':  'nvidia/nemotron-3.5-super-vl-120b-a12b', // early access, multimodal
  'gpt-4o-mini':        'nvidia/nemotron-3.5-lightning-30b-a3b', // fastest 30B MoE
  'nemotron-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',

  // --- StepFun ---
  'step-5':             'stepfun-ai/step-5-preview',          // new Oct 2026
  'step-flash':         'stepfun-ai/step-5-preview',          // legacy alias

  // --- Google / Meta / Poolside / OpenAI OSS ---
  'gemma':              'google/gemma-4-31b-it',              // 262K ctx
  'muse-glimmer':       'meta/muse-glimmer-30b',              // 131K ctx
  'laguna':             'poolside/laguna-xs-2.1',             // coding-focused, 262K ctx
  'claude-3-sonnet':    'openai/gpt-oss-20b',                 // only the 20B is left
  'gpt-oss':            'openai/gpt-oss-20b',

  // --- Legacy Mistral aliases (medium/small no longer listed) ---
  'mistral-medium':     'mistralai/mistral-large',
  'mistral-small':      'nvidia/nemotron-3.5-lightning-30b-a3b',
  'gemini-pro':         'z-ai/glm-5.3',
  'inkling':            'z-ai/glm-5.3',                       // legacy alias
};

// 🔄 FALLBACK CHAIN — tried in order when the requested model is rate-limited / overloaded / gone.
// Entries not present in the live /v1/models list are skipped automatically.
const FALLBACK_CHAIN = [
  'z-ai/glm-5.3',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'deepseek-ai/deepseek-v4.1-flash',
  'z-ai/glm-5.3-flash',
  'nvidia/nemotron-3.5-lightning-30b-a3b',
];

// HTTP statuses that trigger a fallback to the next model
const FALLBACK_STATUSES = new Set([404, 410, 429, 500, 502, 503, 504]);

// 📡 LIVE MODEL LIST — NVIDIA adds/removes models without notice, so ask the API what exists.
let liveModels = null;      // Set<string> | null (null = not loaded yet / fetch failed)
let liveModelsAt = null;

async function refreshLiveModels() {
  try {
    const headers = NIM_API_KEY ? { Authorization: `Bearer ${NIM_API_KEY}` } : {};
    const { data } = await axios.get(`${NIM_API_BASE}/models`, { headers, timeout: 15000 });
    const ids = (data?.data || []).map(m => m.id).filter(Boolean);
    if (ids.length > 0) {
      liveModels = new Set(ids);
      liveModelsAt = new Date().toISOString();
      const dead = Object.values(MODEL_MAPPING).filter((v, i, a) => a.indexOf(v) === i && !liveModels.has(v));
      console.log(`📡 Loaded ${ids.length} live NIM models`);
      if (dead.length) {
        console.warn(`⚠️  Mapped but not in public model list (may still work if gated, or may be gone): ${dead.join(', ')}`);
      }
    }
  } catch (e) {
    console.warn(`⚠️  Could not refresh NIM model list: ${e.message} (using static lists)`);
  }
}

const isLive = (id) => liveModels === null || liveModels.has(id);

// 🛡️ ROLEPLAY GUARD - Injected into every request to prevent the model from speaking as the user
const RP_GUARD_INSTRUCTION = `You are ONLY the character described in the system prompt or conversation. Follow these rules strictly:
- You ONLY speak, act, and think as the character. You do NEVER write or generate any dialogue, actions, or thoughts for the user or any other character that the user is playing.
- Do NOT use labels like "User:", "Human:", "You:" or any prefix to simulate the user's side of the conversation.
- Do NOT continue the conversation by inventing what the user says or does next.
- Stop your response immediately after your character's turn ends.
- If you feel the scene needs a reaction from the user, end your response and wait.`;

// 🛡️ ROLEPLAY GUARD - Strips any text where the model broke character and started writing as the user
function stripUserBreakout(text) {
  const lines = text.split('\n');
  const cleaned = [];
  let dropping = false;

  const userLabels = [
    /^(User|Human|You|Me|Player)\s*[:：]/i,
    /^---+\s*$/,
    /^\*{0,3}\s*(User|Human|You|Me|Player)\s*\*{0,3}\s*[:：]/i
  ];

  for (const line of lines) {
    const trimmed = line.trim();
    if (userLabels.some(pattern => pattern.test(trimmed))) {
      dropping = true;
      continue;
    }
    if (dropping) {
      if (trimmed === '') continue;
      if (trimmed.startsWith('*')) {
        dropping = false;
        cleaned.push(line);
      }
      continue;
    }
    cleaned.push(line);
  }

  const result = cleaned.join('\n');
  const lastUserLabel = result.search(/\n(?:User|Human|You|Me|Player)\s*[:：]/i);
  if (lastUserLabel !== -1) {
    return result.substring(0, lastUserLabel).trimEnd();
  }
  return result.trimEnd();
}

// 🎨 THINKING-CAPABLE MODELS
const THINKING_MODELS = [
  'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-ai/deepseek-v4.1-flash',
  'moonshotai/kimi-k3',
  'moonshotai/kimi-k2.6',
  'z-ai/glm-5.3',
  'z-ai/glm-5.3-flash',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'nvidia/nemotron-3-super-120b-a12b',
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'stepfun-ai/step-5-preview',
];

// Adds the right "thinking on" switch for each model family (NIM reads chat_template_kwargs)
function applyThinkingMode(nimRequest) {
  const m = nimRequest.model;
  if (!ENABLE_THINKING_MODE || !THINKING_MODELS.includes(m)) return;
  if (m.includes('deepseek') || m.includes('kimi')) {
    nimRequest.chat_template_kwargs = { thinking: true };
  } else if (m.includes('glm') || m.includes('nemotron') || m.includes('step')) {
    nimRequest.chat_template_kwargs = { enable_thinking: true };
  }
}

// 🔄 Helper: make a NIM request with automatic fallback (429 / 5xx / 404 / 410 / timeout)
async function makeNimRequest(nimRequest, stream) {
  const fallbacks = FALLBACK_CHAIN.filter(m => m !== nimRequest.model && isLive(m));
  let primary = [nimRequest.model];
  if (!isLive(nimRequest.model) && !ALLOW_UNLISTED_MODELS) {
    console.warn(`⏭️  ${nimRequest.model} is not in NVIDIA's live model list — skipping to fallbacks`);
    primary = [];
  }
  const modelsToTry = [...primary, ...fallbacks].slice(0, MAX_FALLBACK_ATTEMPTS);

  for (let i = 0; i < modelsToTry.length; i++) {
    const modelAttempt = modelsToTry[i];
    const attemptRequest = { ...nimRequest, model: modelAttempt };

    // chat_template_kwargs are model-specific — recompute for the fallback model
    delete attemptRequest.chat_template_kwargs;
    applyThinkingMode(attemptRequest);

    const startedAt = Date.now();
    console.log(`➡️  Trying ${modelAttempt} (attempt ${i + 1}/${modelsToTry.length})`);

    try {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, attemptRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: UPSTREAM_TIMEOUT_MS
      });

      response._usedModel = modelAttempt;
      console.log(`✅ ${modelAttempt} responded in ${Date.now() - startedAt}ms`);
      return response;

    } catch (err) {
      const status = err.response?.status;
      const isTimeout = err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT';
      const isLast = i === modelsToTry.length - 1;

      // Free the socket if upstream sent an error body as a stream
      err.response?.data?.destroy?.();

      if (FALLBACK_STATUSES.has(status) || isTimeout) {
        const why = isTimeout ? 'timeout' : status;
        console.warn(`⚠️  ${why} on ${modelAttempt} after ${Date.now() - startedAt}ms — ${isLast ? 'no more fallbacks' : `trying ${modelsToTry[i + 1]}`}`);
        if (isLast) throw err;
        continue;
      }

      throw err;
    }
  }
}

// Resolve whatever model name Janitor AI sends into a real NIM model id
function resolveModel(requested) {
  if (!requested) return 'z-ai/glm-5.3';
  if (MODEL_MAPPING[requested]) return MODEL_MAPPING[requested];

  // Full NIM ids (e.g. "z-ai/glm-5.3") pass straight through — no probe request needed
  if (liveModels?.has(requested) || requested.includes('/')) return requested;

  const modelLower = requested.toLowerCase();
  if (modelLower.includes('gpt-4') || modelLower.includes('opus') || modelLower.includes('405b')) {
    return 'z-ai/glm-5.3';
  } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
    return 'z-ai/glm-5.3';
  }
  return 'z-ai/glm-5.3-flash'; // fast, reliable default
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy (Janitor AI Optimized)',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_configured: !!NIM_API_KEY,
    available_models: Object.keys(MODEL_MAPPING).length,
    live_nim_models: liveModels ? liveModels.size : null,
    live_models_refreshed: liveModelsAt,
    optimized_for: 'Janitor AI',
    last_updated: 'October 2026'
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    version: '3.0',
    optimized_for: 'Janitor AI',
    status: 'running',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions'
    },
    featured_models: {
      best_quality:   'gpt-4 → z-ai/glm-5.3 (1M ctx)',
      best_roleplay:  'glm-5.3 / gpt-4o → z-ai/glm-5.3 (1M ctx)',
      newest_big:     'kimi-k3 → moonshotai/kimi-k3 (often rate limited)',
      most_popular:   'gpt-3.5-turbo → nemotron-3-ultra-550b',
      fastest:        'gpt-4o-mini → nemotron-3.5-lightning-30b | glm-flash → glm-5.3-flash',
      tip:            'You can also send any full NIM id, e.g. "z-ai/glm-5.3"'
    }
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy',
    nim_model: MODEL_MAPPING[model],
    supports_thinking: THINKING_MODELS.includes(MODEL_MAPPING[model])
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY not configured. Please add your NVIDIA API key in Render environment variables.',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    const { model, messages, temperature, max_tokens, stream,
            top_p, frequency_penalty, presence_penalty, stop, seed } = req.body;

    const nimModel = resolveModel(model);

    // 🛡️ ROLEPLAY GUARD - Inject character-only instruction
    const systemIndex = messages.findIndex(m => m.role === 'system');
    if (systemIndex !== -1) {
      messages[systemIndex] = {
        ...messages[systemIndex],
        content: messages[systemIndex].content + '\n\n' + RP_GUARD_INSTRUCTION
      };
    } else {
      messages.unshift({ role: 'system', content: RP_GUARD_INSTRUCTION });
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens || 12000,
      stream: stream || false
    };

    // Pass through sampling params Janitor AI sends (only if provided)
    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    if (stop !== undefined) nimRequest.stop = stop;
    if (seed !== undefined) nimRequest.seed = seed;

    // 🔄 Use fallback-aware request helper (applies thinking mode per model)
    const response = await makeNimRequest(nimRequest, stream || false);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      let contentAccumulator = '';
      let flushedUpTo = 0;
      const LOOKAHEAD = 200;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              if (contentAccumulator.length > flushedUpTo) {
                const remaining = stripUserBreakout(contentAccumulator.substring(flushedUpTo));
                if (remaining.length > 0) {
                  const doneFlush = {
                    choices: [{ delta: { content: remaining }, index: 0 }]
                  };
                  res.write(`data: ${JSON.stringify(doneFlush)}\n\n`);
                }
              }
              res.write(line + '\n\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                // Newer NIM builds emit `reasoning`, older ones `reasoning_content`
                const reasoning = delta.reasoning_content ?? delta.reasoning;
                const content = delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  if (combinedContent) {
                    delta.content = combinedContent;
                  }
                  delete delta.reasoning_content;
                  delete delta.reasoning;
                } else {
                  delta.content = content || '';
                  delete delta.reasoning_content;
                  delete delta.reasoning;
                }

                const chunkText = delta.content || '';
                if (chunkText) {
                  contentAccumulator += chunkText;
                  const filtered = stripUserBreakout(contentAccumulator);
                  const safeEnd = Math.max(flushedUpTo, filtered.length - LOOKAHEAD);
                  if (safeEnd > flushedUpTo) {
                    const toSend = filtered.substring(flushedUpTo, safeEnd);
                    flushedUpTo = safeEnd;
                    delta.content = toSend;
                    res.write(`data: ${JSON.stringify(data)}\n\n`);
                  }
                  return;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          fullContent = stripUserBreakout(fullContent);
          const reasoningText = choice.message?.reasoning_content ?? choice.message?.reasoning;
          if (SHOW_REASONING && reasoningText) {
            fullContent = '<think>\n' + reasoningText + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    let errorMessage = error.message || 'Internal server error';
    if (error.response?.status === 401) {
      errorMessage = 'Invalid NVIDIA API key. Please check your NIM_API_KEY in environment variables.';
    } else if (error.response?.status === 429) {
      errorMessage = 'Rate limited (NVIDIA free tier is ~40 requests/min). Please wait 60 seconds and try again.';
      res.setHeader('Retry-After', error.response?.headers?.['retry-after'] || 60);
    } else if (error.response?.status === 503) {
      errorMessage = 'NIM servers are overloaded. Fallback models also failed. Please try again shortly.';
    } else if (error.response?.status === 404 || error.response?.status === 410) {
      errorMessage = 'Model is no longer available on NIM and all fallbacks failed. Try a different model.';
    } else if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      errorMessage = 'NVIDIA NIM timed out. Please try again or pick a faster model (e.g. glm-flash).';
    } else if (error.response?.data?.detail) {
      errorMessage = error.response.data.detail;
    }

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: errorMessage,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Available endpoints: /health, /v1/models, /v1/chat/completions`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, '0.0.0.0', async () => {
  console.log('═══════════════════════════════════════════════════════');
  console.log('🚀 OpenAI → NVIDIA NIM Proxy (Janitor AI Optimized)');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`📡 Server running on port ${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log(`📋 Models list: http://localhost:${PORT}/v1/models`);
  console.log('');
  console.log('⚙️  Configuration:');
  console.log(`   • Reasoning display: ${SHOW_REASONING ? '✅ ENABLED' : '❌ DISABLED'}`);
  console.log(`   • Thinking mode:     ${ENABLE_THINKING_MODE ? '✅ ENABLED' : '❌ DISABLED'}`);
  console.log(`   • API key:           ${NIM_API_KEY ? '✅ Configured' : '❌ Missing'}`);
  console.log(`   • Max tokens:        12000 (default)`);
  console.log(`   • Fallback triggers: ${[...FALLBACK_STATUSES].join(', ')} + timeout`);
  console.log(`   • Max attempts:      ${MAX_FALLBACK_ATTEMPTS}`);
  console.log('');
  console.log('🎯 Main Aliases (October 2026):');
  console.log('   • gpt-4            → GLM-5.3');
  console.log('   • gpt-4o           → GLM-5.3');
  console.log('   • glm-flash        → GLM-5.3 Flash');
  console.log('   • deepseek-flash   → DeepSeek V4.1 Flash');
  console.log('   • kimi-k3          → Kimi K3 (often 429 on free tier)');
  console.log('   • gpt-3.5-turbo    → Nemotron 3 Ultra 550B');
  console.log('   • gpt-4o-mini      → Nemotron 3.5 Lightning 30B');
  console.log('   • step-5           → Step-5 Preview');
  console.log('');
  console.log('🔄 Fallback Chain:');
  FALLBACK_CHAIN.forEach((m, i) => console.log(`   ${i + 1}. ${m}`));
  console.log('═══════════════════════════════════════════════════════');

  // Discover what NIM actually serves right now, then refresh every 30 minutes
  await refreshLiveModels();
  setInterval(refreshLiveModels, 30 * 60 * 1000).unref();
});
