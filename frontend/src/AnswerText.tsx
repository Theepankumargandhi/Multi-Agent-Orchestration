export type Citation = { source: string; page: number };
export type Segment =
  | { kind: 'text'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'link'; text: string; url: string }
  | { kind: 'cite'; citations: Citation[]; raw: string };

// Models cite documents as "【source=report.pdf, page=3】" or "【report.pdf, page 3】".
const TOKEN =
  /【([^】\n]{1,300})】|\[([^\]\n]{1,200})\]\((https?:\/\/[^\s)]{1,2000})\)|\*\*([^*\n]{1,500})\*\*|`([^`\n]{1,300})`/g;
const CITATION = /^\s*(?:source\s*=\s*)?([^\s,;]+)\s*,\s*(?:page|p\.)\s*[=:]?\s*(\d{1,5})\s*$/i;

function parseCitations(inner: string): Citation[] | null {
  const citations: Citation[] = [];
  for (const part of inner.split(';')) {
    const match = CITATION.exec(part);
    if (!match) return null;
    citations.push({ source: match[1], page: Number(match[2]) });
  }
  return citations;
}

/** Split an answer into plain text and the few inline forms CiteRight renders. */
export function parseAnswer(text: string): Segment[] {
  const segments: Segment[] = [];
  const pushText = (value: string) => {
    if (!value) return;
    const last = segments.at(-1);
    if (last?.kind === 'text') last.text += value;
    else segments.push({ kind: 'text', text: value });
  };
  let position = 0;
  for (const match of text.matchAll(TOKEN)) {
    pushText(text.slice(position, match.index));
    position = match.index + match[0].length;
    const [raw, cite, linkText, url, bold, code] = match;
    if (cite !== undefined) {
      const citations = parseCitations(cite);
      if (citations) segments.push({ kind: 'cite', citations, raw });
      else pushText(raw);
    } else if (url !== undefined) segments.push({ kind: 'link', text: linkText, url });
    else if (bold !== undefined) segments.push({ kind: 'bold', text: bold });
    else segments.push({ kind: 'code', text: code });
  }
  pushText(text.slice(position));
  return segments;
}

/** The answer as plain words, for one-line previews. */
export const plainText = (text: string) =>
  parseAnswer(text)
    .map((segment) => (segment.kind === 'cite' ? '' : segment.text))
    .join('');

export const citationLabel = (citation: Citation) =>
  `${citation.source.split('/').at(-1)} · p.${citation.page}`;
export const citationKey = (citation: Citation) => `${citation.source}#${citation.page}`;

export function AnswerText({
  text,
  selected,
  onCite,
}: {
  text: string;
  selected?: string;
  onCite?: (citation: Citation) => void;
}) {
  return (
    <>
      {parseAnswer(text).map((segment, index) => {
        switch (segment.kind) {
          case 'bold':
            return <strong key={index}>{segment.text}</strong>;
          case 'code':
            return <code key={index}>{segment.text}</code>;
          case 'link':
            return (
              <a
                key={index}
                className="cite-link"
                href={segment.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                {segment.text}
              </a>
            );
          case 'cite':
            return segment.citations.map((citation) => (
              <button
                key={`${index}-${citationKey(citation)}`}
                type="button"
                className={`cite-chip ${selected === citationKey(citation) ? 'selected' : ''}`}
                title={`${citation.source}, page ${citation.page}`}
                aria-label={`Show source ${citation.source}, page ${citation.page}`}
                aria-pressed={selected === citationKey(citation)}
                onClick={() => onCite?.(citation)}
              >
                {citationLabel(citation)}
              </button>
            ));
          default:
            return segment.text;
        }
      })}
    </>
  );
}
