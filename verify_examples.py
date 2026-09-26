"""Execute the textbook's actual Python examples and check their local contracts.

Run from this repository after installing requirements.txt. This is not a
benchmark reproduction or a substitute for editorial and browser review.
"""
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
import contextlib
import hashlib
import io
import json
import re

ROOT = Path(__file__).resolve().parent

class Chapter(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.codes, self.headings, self.ids = [], [], []
        self.in_code = self.in_heading = False
        self.code, self.heading = [], []
        self.heading_timestamp = False
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'code' and a.get('class') == 'language-python':
            self.in_code = True
            self.code = []
        if tag == 'h2':
            self.in_heading = True
            self.heading = []
            self.heading_timestamp = False
        if tag == 'a' and self.in_heading and 'youtube.com/watch' in a.get('href', '') and '&t=' in a.get('href', ''):
            self.heading_timestamp = True
    def handle_data(self, text):
        if self.in_code:
            self.code.append(text)
        if self.in_heading:
            self.heading.append(text)
    def handle_endtag(self, tag):
        if tag == 'code' and self.in_code:
            self.codes.append(''.join(self.code))
            self.in_code = False
        if tag == 'h2' and self.in_heading:
            self.headings.append((''.join(self.heading), self.heading_timestamp))
            self.in_heading = False

def body(source):
    start, end = '<!-- BOOK-CONTENT-START -->', '<!-- BOOK-CONTENT-END -->'
    if start in source:
        result = source.split(start, 1)[1].split(end, 1)[0]
        guard = re.search(r'name="book-content-sha256" content="([a-f0-9]+)"', source)
        if guard:
            assert hashlib.sha256(result.encode()).hexdigest() == guard[1], 'authored-content digest'
        return result
    return source

def main():
    records = []
    chapters = sorted(ROOT.glob('lecture-[0-9][0-9]-*.html'))
    assert len(chapters) == 18
    manifest = json.loads((ROOT/'source-manifest.json').read_text(encoding='utf-8-sig'))
    durations = {row['id']: row['duration'] for row in manifest['lectures']}
    for chapter in chapters:
        source = body(chapter.read_bytes().decode('utf-8-sig'))
        parsed = Chapter(source)
        assert len(parsed.ids) == len(set(parsed.ids)), chapter.name
        assert parsed.codes, f'{chapter.name}: no declared executable Python'
        outputs = []
        for code in parsed.codes:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(code, chapter.name, 'exec'), {})
            outputs.append(output.getvalue().strip())
        for op, cl in [(r'\(', r'\)'), (r'\[', r'\]')]:
            assert source.count(op) == source.count(cl), (chapter.name, op)
        math = re.findall(r'\\\((.*?)\\\)|\\\[(.*?)\\\]', source, re.S)
        for pair in math:
            text = unescape(pair[0] or pair[1])
            level = 0
            for char in text:
                level += (char == '{') - (char == '}')
                assert level >= 0, (chapter.name, text)
            assert level == 0, (chapter.name, text)
        meta = json.loads(chapter.with_suffix('.meta.json').read_text(encoding='utf-8-sig'))
        for row in meta['sections']:
            if row.get('start') is not None:
                assert 0 <= row['start'] <= row['end'] <= durations[meta['video_id']], (chapter.name, row)
        missing = [title for title, stamped in parsed.headings if not stamped]
        records.append({'chapter': chapter.name, 'embedded_python_blocks_executed': len(parsed.codes),
                        'python_output': outputs, 'math_segments': len(math),
                        'math_delimiters_and_braces': 'passed', 'source_intervals': 'passed',
                        'timestamped_major_sections': sum(stamped for _, stamped in parsed.headings),
                        'unstamped_teaching_sections_for_editorial_review': missing,
                        'body_sha256': hashlib.sha256(source.encode()).hexdigest()})
    from numerical_lab import check_all
    with contextlib.redirect_stdout(io.StringIO()) as lab_output:
        result = check_all()
    report = {'status': 'passed', 'chapters': 18, 'chapter_checks': records,
              'cumulative_lab': result, 'cumulative_lab_stdout': lab_output.getvalue().strip(),
              'boundary': 'Small deterministic fixtures and finite differences; not paper benchmark reproduction.'}
    (ROOT/'numerical-validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"PASS: {len(chapters)} chapters, {sum(r['embedded_python_blocks_executed'] for r in records)} embedded Python blocks.")
    for row in records:
        print(row['chapter'], row['unstamped_teaching_sections_for_editorial_review'])

if __name__ == '__main__':
    main()

