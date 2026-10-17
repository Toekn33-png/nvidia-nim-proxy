
// Janitor AI proxy - NVIDIA NIM + OpenRouter
// Dependencies: express, axios, cors

const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();

app.use(cors());
app.use(express.json({ limit: '5mb' }));

const PORT = process.env.PORT || 3000;

const PROXY_API_KEY = process.env.PROXY_API_KEY;
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || '';
const ROLEPLAY_GUARD = process.env.ROLEPLAY_GUARD !== 'false';

const providers = {
  nvidia: {
    url: process.env.NIM_API_BASE ||
      'https://integrate.api.nvidia.com/v1',
    key: process.env.NIM_API_KEY
  },
  openrouter: {
    url: 'https://openrouter.ai/api/v1',
    key: process.env.OPENROUTER_API_KEY
  }
};

// Model selection
// nim: = NVIDIA
// or: = OpenRouter

function selectProvider(requestedModel) {
  let model = String(
    requestedModel || DEFAULT_MODEL || ''
  ).trim();

  if (!model) {
    return {
      error: 'Set a model name in Janitor AI.'
    };
  }

  let name = 'nvidia';

  if (model.startsWith('or:')) {
    name = 'openrouter';
    model = model.slice(3);
  } else if (model.startsWith('openrouter:')) {
    name = 'openrouter';
    model = model.slice('openrouter:'.length);
  } else if (model.startsWith('nim:')) {
    model = model.slice(4);
  }

  if (!model) {
    return { error: 'Model ID cannot be empty.' };
  }

  return {
    name,
    model,
    ...providers[name]
  };
}

// Protect the proxy with a private password.

function authorize(req, res, next) {
  if (!PROXY_API_KEY) {
    return res.status(503).json({
      error: {
        message: 'Configure PROXY_API_KEY in Render first.'
      }
    });
  }

  const bearer = (
    req.get('Authorization') || ''
  ).replace(/^Bearer\s+/i, '').trim();

console.log('AUTH DEBUG:', {
  authorizationPresent: Boolean(req.get('Authorization')),
  bearerPrefix: /^Bearer\s+/i.test(req.get('Authorization') || ''),
  xApiKeyPresent: Boolean(req.get('x-api-key')),
  passwordMatches: bearer === PROXY_API_KEY
});

  if (bearer !== PROXY_API_KEY) {
    return res.status(401).json({
      error: {
        message: 'Invalid proxy key. Use PROXY_API_KEY in Janitor AI.'
      }
    });
  }

  next();
}

// Home endpoint

app.get('/', (req, res) => {
  res.json({
    service: 'Janitor AI Proxy',
    status: 'running',
    providers: ['NVIDIA', 'OpenRouter'],
    endpoints: [
      '/health',
      '/v1/models',
      '/v1/chat/completions'
    ]
  });
});

// Health endpoint

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    proxy_key_configured: Boolean(PROXY_API_KEY),
    nvidia_configured: Boolean(providers.nvidia.key),
    openrouter_configured: Boolean(providers.openrouter.key)
  });
});

// List models from provider

app.get('/v1/models', authorize, async (req, res) => {
  const name =
    req.query.provider === 'openrouter'
      ? 'openrouter'
      : 'nvidia';

  const provider = providers[name];

  if (!provider.key) {
    return res.status(503).json({
      error: {
        message: `${name} API key is not configured.`
      }
    });
  }

  try {
    const response = await axios.get(
      `${provider.url}/models`,
      {
        headers: {
          Authorization: `Bearer ${provider.key}`
        },
        timeout: 20000
      }
    );

    const prefix =
      name === 'openrouter' ? 'or:' : 'nim:';

    const listing = response.data;

    if (Array.isArray(listing?.data)) {
      return res.json({
        ...listing,
        data: listing.data.map(item => ({
          ...item,
          id: prefix + item.id
        }))
      });
    }

    res.json(listing);
  } catch (error) {
    const status = error.response?.status || 502;

    res.status(status).json({
      error: {
        message: `Could not fetch ${name} models. HTTP ${status}.`
      }
    });
  }
});

// Roleplay instruction

const GUARD_TEXT =
  "In roleplay, write only characters controlled by the assistant " +
  "and scene narration. Never write the user character's dialogue, " +
  "actions, thoughts, or decisions. Allow the user to respond.";

function messagesForRoleplay(messages) {
  const result = messages.map(message => ({
    ...message
  }));

  if (!ROLEPLAY_GUARD) {
    return result;
  }

  const systemIndex = result.findIndex(
    message =>
      message.role === 'system' &&
      typeof message.content === 'string'
  );

  if (systemIndex >= 0) {
    result[systemIndex].content +=
      '\n\n' + GUARD_TEXT;
  } else {
    result.unshift({
      role: 'system',
      content: GUARD_TEXT
    });
  }

  return result;
}

// Extract actual provider errors

async function getUpstreamError(error) {
  let data = error.response?.data;

  if (data && typeof data.on === 'function') {
    let message = '';

    try {
      for await (const chunk of data) {
        message += chunk.toString();

        if (message.length > 12000) {
          break;
        }
      }
    } catch (_) {}

    data = message;
  }

  if (typeof data === 'string') {
    try {
      data = JSON.parse(data);
    } catch (_) {
      return data.slice(0, 500);
    }
  }

  const value =
    data?.error?.message ||
    data?.detail ||
    error.message ||
    'Unknown error';

  return typeof value === 'string'
    ? value.slice(0, 500)
    : JSON.stringify(value).slice(0, 500);
}

// Main chat completions endpoint

app.post(
  '/v1/chat/completions',
  authorize,
  async (req, res) => {

    const selected = selectProvider(req.body?.model);

    if (selected.error) {
      return res.status(400).json({
        error: {
          message: selected.error
        }
      });
    }

    if (!selected.key) {
      return res.status(503).json({
        error: {
          message: `Missing ${selected.name} API key in Render.`
        }
      });
    }

    if (
      !Array.isArray(req.body.messages) ||
      req.body.messages.length === 0
    ) {
      return res.status(400).json({
        error: {
          message: 'Messages must be a nonempty array.'
        }
      });
    }

    const streaming = req.body.stream === true;

    const configuredCap = Number(
      process.env.MAX_OUTPUT_TOKENS || 2200
    );

    const cap =
      Number.isFinite(configuredCap) &&
      configuredCap > 0
        ? Math.floor(configuredCap)
        : 2200;

    const requestedTokens = Number(
      req.body.max_tokens
    );

    const tokens =
      Number.isFinite(requestedTokens) &&
      requestedTokens > 0
        ? requestedTokens
        : 1600;

    const requestData = {
      model: selected.model,
      messages: messagesForRoleplay(
        req.body.messages
      ),
      stream: streaming,
      max_tokens: Math.min(
        Math.floor(tokens),
        cap
      )
    };

    if (req.body.temperature != null) {
      requestData.temperature =
        req.body.temperature;
    }

    if (req.body.top_p != null) {
      requestData.top_p = req.body.top_p;
    }

    if (req.body.stop != null) {
      requestData.stop = req.body.stop;
    }

    try {
      const upstream = await axios.post(
        `${selected.url}/chat/completions`,
        requestData,
        {
          headers: {
            Authorization: `Bearer ${selected.key}`,
            'Content-Type': 'application/json'
          },
          timeout: 180000,
          responseType: streaming
            ? 'stream'
            : 'json'
        }
      );

      // Do not switch models automatically.

      if (streaming) {
        res.status(200);

        res.setHeader(
          'Content-Type',
          'text/event-stream; charset=utf-8'
        );

        res.setHeader(
          'Cache-Control',
          'no-cache, no-transform'
        );

        res.setHeader(
          'X-Accel-Buffering',
          'no'
        );

        upstream.data.on('error', err => {
          console.error(
            'Upstream stream interrupted:',
            err.message
          );

          res.end();
        });

        upstream.data.pipe(res);
      } else {
        res.json(upstream.data);
      }

    } catch (error) {
      const status =
        error.response?.status || 502;

      const detail =
        await getUpstreamError(error);

      console.error(
        `Upstream ${selected.name} (${selected.model}) HTTP ${status}: ${detail}`
      );

      if (!res.headersSent) {
        res.status(status).json({
          error: {
            message:
              `${selected.name}/${selected.model}: ${detail}`,
            code: status
          }
        });
      } else {
        res.end();
      }
    }
  }
);

// Unsupported routes

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint not found.'
    }
  });
});

// Start server

app.listen(PORT, '0.0.0.0', () => {
  console.log(
    `Janitor AI proxy listening on port ${PORT}`
  );

  console.log(
    `NVIDIA configured: ${Boolean(providers.nvidia.key)}`
  );

  console.log(
    `OpenRouter configured: ${Boolean(providers.openrouter.key)}`
  );

  console.log(
    `Proxy password configured: ${Boolean(PROXY_API_KEY)}`
  );
});
