import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClient } from './api';
import { Research } from './Research';
import { Badge, Icon, Notice, errorText, isAbort } from './components';
import type { Capabilities, Session } from './types';

function Brand() {
  return (
    <div className="brand">
      <span className="brand-symbol">
        C<span>R</span>
      </span>
      <div>
        CiteRight<small>CITED ANSWERS</small>
      </div>
    </div>
  );
}
function Login({
  capabilities,
  onSession,
  status,
}: {
  capabilities: Capabilities | null;
  onSession: (session: Session) => void;
  status: string;
}) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [user, setUser] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const api = useMemo(() => new ApiClient(), []);
  useEffect(() => () => api.dispose(), [api]);
  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const session = await api.authenticate(mode, user.trim(), password);
      if (
        typeof session.access_token !== 'string' ||
        !session.access_token ||
        !Number.isFinite(session.expires_in) ||
        session.expires_in <= 0
      )
        throw new Error('Invalid authentication response.');
      setPassword('');
      onSession(session);
    } catch (e) {
      if (!isAbort(e)) setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="login-layout">
      <section className="login-story">
        <Brand />
        <div>
          <p className="eyebrow">ANSWERS YOU CAN CHECK</p>
          <h1>
            Ask anything.
            <br />
            See the source.
          </h1>
          <p className="login-lead">
            CiteRight answers from your documents and the web,
            <br />
            cites where every claim came from, and says so when it can't.
          </p>
          <div className="story-steps">
            <span>
              <Icon name="research" /> Ask a question
            </span>
            <span>
              <Icon name="shield" /> Claims are checked against sources
            </span>
            <span>
              <Icon name="arrow" /> Follow each citation
            </span>
          </div>
        </div>
        <p className="story-footer">Your provider keys stay on the server.</p>
      </section>
      <section className="login-form-area">
        <form onSubmit={submit} className="login-card">
          <Badge tone={capabilities ? 'green' : ''}>
            {capabilities ? 'API reachable' : 'API unavailable'}
          </Badge>
          <h2>Welcome to CiteRight</h2>
          <p className="muted">Sign in to see your conversations.</p>
          <div className="segmented" aria-label="Authentication mode">
            <button
              type="button"
              aria-pressed={mode === 'login'}
              onClick={() => {
                setMode('login');
                setError('');
              }}
            >
              Sign in
            </button>
            <button
              type="button"
              aria-pressed={mode === 'register'}
              onClick={() => {
                setMode('register');
                setError('');
              }}
            >
              Create account
            </button>
          </div>
          {status && <Notice>{status}</Notice>}
          {error && <Notice error>{error}</Notice>}
          {capabilities?.features.user_authentication === false && (
            <Notice error>CiteRight requires ENABLE_USER_AUTH=true on the backend.</Notice>
          )}
          <label>
            User ID
            <input
              name="username"
              value={user}
              onChange={(e) => setUser(e.target.value)}
              autoComplete="username"
              pattern="[A-Za-z0-9._-]{3,64}"
              minLength={3}
              maxLength={64}
              required
              disabled={busy}
              placeholder="your-user-id"
            />
          </label>
          <label>
            Password
            <input
              name="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              minLength={8}
              maxLength={256}
              required
              disabled={busy}
            />
          </label>
          <button
            className="button full"
            disabled={busy || !capabilities || capabilities.features.user_authentication === false}
          >
            {busy
              ? 'Connecting…'
              : mode === 'login'
                ? 'Sign in to CiteRight'
                : 'Create your account'}
            <Icon name="arrow" size={17} />
          </button>
          <p className="small-text muted">
            Session tokens stay in memory, not browser storage. Refreshing the page signs you out.
          </p>
        </form>
      </section>
    </main>
  );
}

function Workspace({
  session,
  capabilities,
  logout,
}: {
  session: Session;
  capabilities: Capabilities;
  logout: (message?: string) => void;
}) {
  const api = useMemo(
    () => new ApiClient(session.access_token, () => logout('Your session expired. Sign in again.')),
    [session],
  );
  useEffect(() => {
    const timer = setTimeout(
      () => logout('Your session expired. Sign in again.'),
      Math.min(session.expires_in * 1000, 2_147_483_647),
    );
    return () => {
      clearTimeout(timer);
      api.dispose();
    };
  }, [api, session]);
  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to conversation
      </a>
      <aside className="sidebar">
        <Brand />
        <div className="sidebar-note">
          <Icon name="shield" size={19} />
          <strong>Checked, not guessed.</strong>
          <p>
            Every answer is checked against its sources before you see it. Unsupported claims are
            removed.
          </p>
        </div>
        <div className="user-block">
          <span className="avatar">{session.user_id.slice(0, 2).toUpperCase()}</span>
          <div>
            <strong>{session.user_id}</strong>
            <small>In-memory session</small>
          </div>
          <button
            className="icon-button"
            aria-label="Sign out"
            title="Sign out"
            onClick={() => logout()}
          >
            <Icon name="logout" size={18} />
          </button>
        </div>
      </aside>
      <div className="main-shell">
        <main id="workspace" className="workspace-main">
          <Research api={api} capabilities={capabilities} />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [status, setStatus] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const api = new ApiClient();
    void api
      .request<Capabilities>('/capabilities')
      .then((value) => {
        setCapabilities(value);
        setStatus('');
      })
      .catch(() =>
        setStatus('The API is unreachable. Start the backend and check the local proxy target.'),
      );
    return () => api.dispose();
  }, [attempt]);
  function logout(message = '') {
    setSession(null);
    setStatus(message);
  }
  if (session && capabilities)
    return (
      <Workspace
        key={session.user_id}
        session={session}
        capabilities={capabilities}
        logout={logout}
      />
    );
  return (
    <>
      <Login capabilities={capabilities} onSession={setSession} status={status} />
      {!capabilities && (
        <button
          className="connection-retry button secondary"
          onClick={() => setAttempt((value) => value + 1)}
        >
          Retry API connection
        </button>
      )}
    </>
  );
}
