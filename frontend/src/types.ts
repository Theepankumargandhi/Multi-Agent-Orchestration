export type ChatMessage = {
  type: 'human' | 'ai' | 'tool';
  content: string;
  run_id?: string | null;
  tool_call_id?: string | null;
  tool_calls?: { id: string; name: string; args: unknown }[];
};
export type Capabilities = {
  models: { id: string; label: string }[];
  features: Record<string, boolean>;
};
export type Session = { access_token: string; user_id: string; expires_in: number };
export type WorkbenchConfig = {
  version: string;
  evidence_mode?: string;
  identity: string;
  code_enabled: boolean;
  repositories: string[];
  coding_models: string[];
  workflow: string;
  test_command: string[];
  integrations: Record<string, string>;
};
export type Thread = { thread_id: string; last_message_preview?: string; message_count?: number };
export type TaskSummary = {
  task_id: string;
  status: string;
  repository: string;
  issue_preview: string;
  model: string;
  updated_at: string;
};
export type TestResult = { exit_code: number; stdout: string; stderr: string; timed_out?: boolean };
export type TaskResult = {
  summary: string;
  changed_files: string[];
  model_calls: number;
  tool_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  usage_accounting_complete?: boolean | null;
  estimated_cost_usd?: number | null;
  baseline_test?: TestResult | null;
  final_test?: TestResult | null;
  verification?: { final_decision: string; blocking_reasons: string[] } | null;
  observations?: {
    iteration: number;
    stage?: string;
    action: string;
    ok: boolean;
    summary: string;
    output?: string;
  }[];
};
export type CodeTask = {
  task_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  error?: string;
  request: { repository: string; issue: string; model: string };
  result?: TaskResult | null;
};
export type TaskEvent = {
  occurred_at: string;
  from_status: string;
  to_status: string;
  actor: string;
  detail?: unknown;
};
