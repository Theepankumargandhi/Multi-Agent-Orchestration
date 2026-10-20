import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClient } from './api';
import { Coding } from './Coding';
import { Documents } from './Documents';
import { Research } from './Research';
import { Badge, Empty, Icon, Notice, PageHeading, errorText, isAbort } from './components';
import type { Capabilities, Session, WorkbenchConfig } from './types';

type Page = 'overview' | 'research' | 'coding' | 'documents' | 'evaluations';
const navigation: { id: Page; label: string; detail?: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'research', label: 'Research' },
  { id: 'coding', label: 'Coding' },
  { id: 'documents', label: 'Documents' },
  { id: 'evaluations', label: 'Evaluations', detail: 'Reports' },
];
function Brand() {
  return (
    <div className="brand">
      <span className="brand-symbol">
        A<span>F</span>
      </span>
      <div>
        AgentForge<small>WORKBENCH</small>
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
          <p className="eyebrow">FROM ANSWER TO EVIDENCE</p>
          <h1>
            Give your agents
            <br />a workbench.
          </h1>
          <p className="login-lead">
            Research with sources. Repair with tests.
            <br />
            Keep the evidence in view.
          </p>
          <div className="story-steps">
            <span>
              <Icon name="research" /> Ask a clear question
            </span>
            <span>
              <Icon name="coding" /> Observe the execution
            </span>
            <span>
              <Icon name="shield" /> Review before release
            </span>
          </div>
        </div>
        <p className="story-footer">
          A reliability-first AI engineering workspace.
          <br />
          Local integration · explicit model access · human review
        </p>
      </section>
      <section className="login-form-area">
        <form onSubmit={submit} className="login-card">
          <Badge tone={capabilities ? 'green' : ''}>
            {capabilities ? 'API reachable' : 'API unavailable'}
          </Badge>
          <h2>Welcome to your workspace</h2>
          <p className="muted">Sign in to access your conversations and coding tasks.</p>
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
            <Notice error>The Workbench requires ENABLE_USER_AUTH=true on the backend.</Notice>
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
                ? 'Sign in to Workbench'
                : 'Create your account'}
            <Icon name="arrow" size={17} />
          </button>
          <p className="small-text muted">
            Session tokens stay in memory—not browser storage. Refreshing the page signs you out.
            Provider API keys belong only on the server.
          </p>
        </form>
        <span className="login-bottom">AgentForge / Workbench 0.1</span>
      </section>
    </main>
  );
}

function Overview({
  config,
  capabilities,
  navigate,
}: {
  config: WorkbenchConfig;
  capabilities: Capabilities;
  navigate: (page: Page) => void;
}) {
  const enabled = Object.values(capabilities.features).filter(Boolean).length;
  return (
    <>
      <PageHeading eyebrow="WORKSPACE OVERVIEW" title="Make the work inspectable.">
        One place to research, inspect agent execution, and review proposed changes.
      </PageHeading>
      <section className="overview-hero">
        <div>
          <Badge tone="light">WORKBENCH / 0.11</Badge>
          <h2>
            Useful agents.
            <br />
            Visible evidence.
          </h2>
          <p>
            Start a conversation or submit a controlled repair.
            <br />
            Your backend remains the authority on what can run.
          </p>
          <div className="button-row">
            <button className="button white" onClick={() => navigate('research')}>
              Open research <Icon name="arrow" size={17} />
            </button>
            <button className="button outline" onClick={() => navigate('coding')}>
              Inspect coding tasks
            </button>
          </div>
        </div>
        <div className="evidence-stack" aria-hidden="true">
          <div className="stack-card">
            <span>01 / RETRIEVE</span>
            <strong>Relevant sources</strong>
            <i />
          </div>
          <div className="stack-card">
            <span>02 / EXECUTE</span>
            <strong>Bounded tools & tests</strong>
            <i />
          </div>
          <div className="stack-card">
            <span>03 / REVIEW</span>
            <strong>Evidence before release</strong>
            <b>HUMAN IN THE LOOP</b>
          </div>
        </div>
      </section>
      <div className="overview-metrics">
        <div>
          <small>Configured models</small>
          <strong>{capabilities.models.length}</strong>
          <span>Configuration, not a live provider check</span>
        </div>
        <div>
          <small>Browser-enabled repositories</small>
          <strong>{config.repositories.length}</strong>
          <span>Explicit operator allowlist</span>
        </div>
        <div>
          <small>Backend feature flags on</small>
          <strong>{enabled}</strong>
          <span>Not a readiness or quality score</span>
        </div>
      </div>
      <div className="section-title">
        <h2>Choose a workspace</h2>
        <span className="muted small-text">Availability is explicit.</span>
      </div>
      <div className="workspace-cards">
        {[
          {
            id: 'research' as const,
            title: 'Research with evidence',
            text: 'Streaming chat, saved conversations, web approvals and actual tool events.',
            status: capabilities.models.length ? 'Connected' : 'Configure a provider',
          },
          {
            id: 'coding' as const,
            title: 'Repair with verification',
            text: 'Owner-scoped tasks, lifecycle history, test results and gated patch access.',
            status: config.code_enabled ? 'Connected' : 'Requires configuration',
          },
          {
            id: 'documents' as const,
            title: 'Understand your documents',
            text: 'Private uploads, isolated OCR, literal citations, source regions, table arithmetic and owner review.',
            status:
              config.integrations.documents === 'local_extractive'
                ? 'Connected'
                : 'Requires configuration',
          },
          {
            id: 'evaluations' as const,
            title: 'Measure what changed',
            text: 'Existing report dashboards remain available separately. No benchmark runs are launched here.',
            status: 'Separate dashboards',
          },
        ].map((item) => (
          <button className="workspace-card" key={item.id} onClick={() => navigate(item.id)}>
            <div>
              <span className="card-icon">
                <Icon name={item.id} size={23} />
              </span>
              <Icon name="arrow" size={18} />
            </div>
            <h3>{item.title}</h3>
            <p>{item.text}</p>
            <Badge tone={item.status === 'Connected' ? 'green' : ''}>{item.status}</Badge>
          </button>
        ))}
      </div>
    </>
  );
}
function Integration() {
  return (
    <>
      <PageHeading
        eyebrow="INTEGRATION STATUS"
        title="Evaluation evidence"
        actions={<Badge>Separate dashboards</Badge>}
      >
        Existing evaluation tooling remains available. This release does not expose private report
        directories.
      </PageHeading>
      <section className="panel integration-panel">
        <Empty title="Reports are not model-quality guarantees.">
          Use the existing local Streamlit dashboards to inspect registered evaluations.
          Authenticated report access and independent review intake are a later integration.
        </Empty>
      </section>
    </>
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
  const [config, setConfig] = useState<WorkbenchConfig | null>(null);
  const [error, setError] = useState('');
  const [page, setPage] = useState<Page>('overview');
  const [allowCalls, setAllowCalls] = useState(false);
  const [configAttempt, setConfigAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    void api
      .request<WorkbenchConfig>('/workbench/config', { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setConfig(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, configAttempt]);
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
        Skip to workspace
      </a>
      <aside className="sidebar">
        <Brand />
        <div className="sidebar-section">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              key={item.id}
              aria-label={`${item.label}${item.detail ? ` ${item.detail}` : ''}`}
              aria-current={page === item.id ? 'page' : undefined}
              onClick={() => setPage(item.id)}
            >
              <Icon name={item.id} size={19} />
              <span>{item.label}</span>
              {item.detail && <small>{item.detail}</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <Icon name="shield" size={19} />
          <strong>Evidence, not assumptions.</strong>
          <p>Runtime controls stay on the backend. No workflow runs on page load.</p>
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
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace</span>
            <span className="slash">/</span>
            <strong>{navigation.find((item) => item.id === page)?.label}</strong>
          </div>
          <div className="connection">
            <span className="tiny-dot" /> FastAPI connected <span className="topbar-divider" />{' '}
            Local workbench
          </div>
        </header>
        <main id="workspace" className="workspace-main">
          {config?.evidence_mode === 'authored_fixture' && (
            <Notice>
              AUTHORED TEST FIXTURE — synthetic responses and lifecycle events; no model or sandbox
              execution. This is not a live-agent result.
            </Notice>
          )}
          <div className="consent-strip">
            <div>
              <Icon name="shield" size={17} />
              <span>
                New model requests are{' '}
                <strong>{allowCalls ? 'enabled for this session' : 'paused'}</strong>. Reading
                history does not call a model.
              </span>
            </div>
            <label className="check-label">
              <input
                type="checkbox"
                checked={allowCalls}
                onChange={(e) => setAllowCalls(e.target.checked)}
              />{' '}
              Allow model requests
            </label>
          </div>
          {!config ? (
            <section className="panel">
              <Empty
                title={error ? 'Cannot load workspace configuration' : 'Connecting your workspace…'}
              >
                {error || 'Loading authorized capabilities. No model requests are being made.'}
              </Empty>
              {error && (
                <button
                  className="button secondary"
                  onClick={() => {
                    setError('');
                    setConfigAttempt((value) => value + 1);
                  }}
                >
                  Retry connection
                </button>
              )}
            </section>
          ) : page === 'overview' ? (
            <Overview config={config} capabilities={capabilities} navigate={setPage} />
          ) : page === 'research' ? (
            <Research api={api} capabilities={capabilities} allowCalls={allowCalls} />
          ) : page === 'coding' ? (
            <Coding api={api} config={config} allowCalls={allowCalls} />
          ) : page === 'documents' ? (
            <Documents api={api} enabled={config.integrations.documents === 'local_extractive'} />
          ) : (
            <Integration />
          )}
          <footer className="workspace-footer">
            <span>AgentForge Workbench 0.11</span>
            <span>Provenance ≠ correctness. Inspect the evidence.</span>
          </footer>
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
