import { describe, expect, it } from 'vitest';
import { citationLabel, parseAnswer, plainText } from '../AnswerText';

describe('parseAnswer', () => {
  it('turns the citation forms models write into citations', () => {
    expect(
      parseAnswer(
        'Sold **4,200** robots【source=report.pdf, page=1】 and more【report.pdf, page 2】.',
      ),
    ).toEqual([
      { kind: 'text', text: 'Sold ' },
      { kind: 'bold', text: '4,200' },
      { kind: 'text', text: ' robots' },
      {
        kind: 'cite',
        citations: [{ source: 'report.pdf', page: 1 }],
        raw: '【source=report.pdf, page=1】',
      },
      { kind: 'text', text: ' and more' },
      {
        kind: 'cite',
        citations: [{ source: 'report.pdf', page: 2 }],
        raw: '【report.pdf, page 2】',
      },
      { kind: 'text', text: '.' },
    ]);
  });
  it('splits several citations in one marker', () => {
    const [segment] = parseAnswer('【source=docs/a.md, page=3; source=b.md, page=12】');
    expect(segment).toMatchObject({
      kind: 'cite',
      citations: [
        { source: 'docs/a.md', page: 3 },
        { source: 'b.md', page: 12 },
      ],
    });
    expect(citationLabel({ source: 'docs/a.md', page: 3 })).toBe('a.md · p.3');
  });
  it('keeps unrecognised markers and unsafe links as plain text', () => {
    expect(parseAnswer('See 【1†L1-L3】 and [x](javascript:alert(1)).')).toEqual([
      { kind: 'text', text: 'See 【1†L1-L3】 and [x](javascript:alert(1)).' },
    ]);
  });
  it('gives previews plain words without markup or citations', () => {
    expect(plainText('Sold **4,200** robots【source=report.pdf, page=1】.')).toBe(
      'Sold 4,200 robots.',
    );
  });
  it('renders web sources as links and inline code', () => {
    expect(parseAnswer('Run `make test`. Sources: [Docs](https://example.com/a)')).toEqual([
      { kind: 'text', text: 'Run ' },
      { kind: 'code', text: 'make test' },
      { kind: 'text', text: '. Sources: ' },
      { kind: 'link', text: 'Docs', url: 'https://example.com/a' },
    ]);
  });
});
