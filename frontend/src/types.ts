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
export type Thread = { thread_id: string; last_message_preview?: string; message_count?: number };
