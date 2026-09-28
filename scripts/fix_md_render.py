#!/usr/bin/env python3
r"""让 markdown 里的公式和加粗在 GitHub 上正确渲染，同时保持 MathJax / KaTeX（Cursor、VS Code 预览）可用。

GitHub 的坑（2026-09 用 /markdown API 和线上页面实测）：
  1. 行内公式开头的 $ 前面必须是空格、行首、'(' 或 '**'；紧贴中文或全角标点（，：（、）不渲染。
     结尾的 $ 后面不能紧跟 ASCII 字母、数字或 '_'。
  2. markdown 的反斜杠转义先作用在公式里：\{ \} \| \, \; \: \! \\ \% \_ \# 会丢掉反斜杠。
     改用 \lbrace \rbrace \Vert \thinspace \thickspace \medspace \negthinspace \cr {\char"XX}。
  3. 公式里的 '*' 和"标点后面的 '_'"（如 \mathbf{x}_t）会被当成强调标记，吃掉下标、拆散公式。
     '*' 改成 \ast；这类 '_' 后面补一个空格（TeX 忽略，markdown 不再把它当强调开头）。
  4. 含 \begin{...} 的 $$...$$ 必须单独成段、放在顶层（不能在段落中间或列表项里）。
  5. 加粗的 ** 紧贴全角标点时闭合不了（**……。**后文），把标点挪到 ** 外面或补空格。

用法：python3 scripts/fix_md_render.py [--dry] [文件...]   # 不给文件就处理 git 跟踪的全部 .md；幂等
"""
import re, sys, subprocess, collections, unicodedata
MAP = {'{': r'\lbrace', '}': r'\rbrace', '|': r'\Vert', ',': r'\thinspace', ';': r'\thickspace',
       ':': r'\medspace', '>': r'\medspace', '!': r'\negthinspace'}
TEXTCMDS = {r'\text', r'\textbf', r'\textit', r'\texttt', r'\textrm', r'\textsf', r'\textup', r'\mbox'}
flags = []

def fix_tex(s, where):
    out = []; i = 0; n = len(s)
    def emit(rep, nxt):
        out.append(rep + (' ' if nxt < n and s[nxt].isalpha() else ''))
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            d = s[i + 1]
            if d.isalpha():
                j = i + 1
                while j < n and s[j].isalpha(): j += 1
                word = s[i:j]
                if word == r'\verb':
                    k = s.index(s[j], j + 1); out.append(s[i:k + 1]); i = k + 1; continue
                if word in TEXTCMDS:
                    k = j
                    while k < n and s[k] == ' ': k += 1
                    if k < n and s[k] == '{':
                        depth = 0; m = k
                        while m < n:
                            if s[m] == '\\': m += 2; continue
                            if s[m] == '{': depth += 1
                            elif s[m] == '}':
                                depth -= 1
                                if depth == 0: break
                            m += 1
                        arg = s[k:m + 1]
                        if re.search(r'\\[^A-Za-z]|[*]', arg[1:-1]): flags.append(f'{where}: text-mode arg needs review: {word}{arg}')
                        out.append(s[i:m + 1]); i = m + 1; continue
                out.append(word); i = j; continue
            if d in MAP:
                emit(MAP[d], i + 2); i += 2; continue
            if d == '\\':
                i += 2
                m = re.match(r'\s*\[[^\]]*\]', s[i:])
                if m: i += m.end()
                emit(r'\cr', i); continue
            if d in '_#%$':
                out.append('{\\char"%02X}' % ord(d)); i += 2; continue
            if d in '%_#&$~^[]()"\'`<=+-./?@':
                if d not in '()[]' and d != '/':
                    flags.append(f'{where}: escape \\{d} left as-is in: {s[:120]}')
            out.append(s[i:i + 2]); i += 2; continue
        if c == '_':
            # markdown: '_' after a non-alphanumeric char and before a non-space can OPEN emphasis and eat
            # subscripts across formulas. A following space makes it non-left-flanking; TeX ignores it.
            prev = ''.join(out)[-1:] if out else ''
            if not prev.isalnum() and i + 1 < n and not s[i + 1].isspace():
                out.append('_ '); i += 1; continue
        if c == '*':
            prev = ''.join(out).rstrip()[-1:] if out else ''
            if prev in ('^', '_'): out.append(r'{\ast}')
            else: emit(r'\ast', i + 1)
            i += 1; continue
        out.append(c); i += 1
    return ''.join(out)

INLINE = re.compile(r'(?<!\\)\$\$((?:\\.|[^$\\])+?)\$\$|(?<![\\$])\$(?![\s$])((?:\\.|[^$\\\n])+?)(?<!\s)\$(?!\$)')
RUN = re.compile(r'[A-Za-z0-9./+\-]+$')
ALLOWED_PRE = lambda t: (t == '' or t[-1].isspace() or t[-1] == '(' or t.endswith('**'))

def fix_line(line, where, stats):
    codes = []
    def prot(m): codes.append(m.group(0)); return f'\x00{len(codes)-1}\x00'
    L = re.sub(r'(`+)(?!`).+?(?<!`)\1(?!`)', prot, line)
    toks = []; pos = 0
    for m in INLINE.finditer(L):
        toks.append(['text', L[pos:m.start()]])
        if m.group(1) is not None: toks.append(['disp', m.group(1)])
        else: toks.append(['math', m.group(2)])
        pos = m.end()
    toks.append(['text', L[pos:]])
    if len(toks) == 1: return line
    # 1) merge $a$/$b$, $a$–$b$, $a$-$b$
    SEP = {'/': '/', '–': r'\text{–}', '-': r'\text{-}'}
    k = 0
    while k + 2 < len(toks):
        if toks[k][0] == 'math' and toks[k+1][0] == 'text' and toks[k+1][1] in SEP and toks[k+2][0] == 'math':
            toks[k][1] = toks[k][1] + SEP[toks[k+1][1]] + toks[k+2][1]; del toks[k+1:k+3]; stats['merged'] += 1
        else: k += 1
    # 2) adjacency
    for k, t in enumerate(toks):
        if t[0] not in ('math', 'disp'): continue
        prev = toks[k-1]
        if t[0] == 'math' and not ALLOWED_PRE(prev[1]):
            p = prev[1][-1]
            m = RUN.search(prev[1])
            if m and (p.isascii() and (p.isalnum() or p == '-')) and m.group(0).strip('./+-') :
                run = m.group(0)
                prev[1] = prev[1][:m.start()]; t[1] = r'\text{' + run + '}' + t[1]; stats['absorbed'] += 1
        if not ALLOWED_PRE(prev[1]):
            prev[1] = prev[1] + ' '; stats['space_before'] += 1
        nxt = toks[k+1] if k + 1 < len(toks) else None
        if nxt and nxt[0] == 'text' and nxt[1][:1] and (nxt[1][0].isascii() and (nxt[1][0].isalnum() or nxt[1][0] == '_')):
            nxt[1] = ' ' + nxt[1]; stats['space_after'] += 1
    # 3) tex fixes
    out = []
    for t in toks:
        if t[0] == 'text': out.append(t[1])
        else:
            new = fix_tex(t[1], where)
            if new != t[1]: stats['tex_fixed'] += 1
            out.append(('$$' if t[0] == 'disp' else '$') + new + ('$$' if t[0] == 'disp' else '$'))
    R = ''.join(out)
    return re.sub(r'\x00(\d+)\x00', lambda m: codes[int(m.group(1))], R)

def fix_file(path, stats):
    lines = open(path).read().split('\n'); out = []; i = 0; fence = None
    while i < len(lines):
        line = lines[i]; body = re.sub(r'^[\s>]*', '', line)
        fm = re.match(r'(`{3,}|~{3,})', body)
        if fence:
            out.append(line)
            if fm and body.startswith(fence) and body.strip() == body.strip()[0] * len(body.strip()): fence = None
            i += 1; continue
        if fm: fence = fm.group(1); out.append(line); i += 1; continue
        # multi-line display block: line starting with $$ having an odd number of $$
        if body.startswith('$$') and body.count('$$') % 2 == 1:
            j = i + 1
            while j < len(lines) and lines[j].count('$$') % 2 == 0: j += 1
            if j < len(lines):
                block = '\n'.join(lines[i:j+1])
                prefix = line[:len(line) - len(body)]
                inner_start = block.index('$$') + 2; inner_end = block.rindex('$$')
                inner = block[inner_start:inner_end]
                new_inner = fix_tex(inner, f'{path}:{i+1}')
                if new_inner != inner: stats['tex_fixed'] += 1
                if prefix.strip() or prefix:   # inside a list / quote: GitHub needs a single line
                    new_inner = ' '.join(x.strip() for x in new_inner.split('\n') if x.strip())
                    out.append(prefix + '$$' + new_inner + '$$' + block[inner_end+2:]); stats['display_flattened'] += 1
                else:
                    out.extend((block[:inner_start] + new_inner + block[inner_end:]).split('\n'))
                i = j + 1; continue
        out.append(fix_line(line, f'{path}:{i+1}', stats)); i += 1
    return '\n'.join(out)



def fix_display_env(files):
    changed = []
    for f in files:
        L = open(f).read().split('\n'); out = []; mod = False
        for i, line in enumerate(L):
            s = line.strip()
            if re.fullmatch(r'\$\$.*\\begin\{.*\$\$', s) and not line.lstrip().startswith('>'):
                if out and out[-1].strip() != '': out.append(''); mod = True
                if line != s: mod = True
                out.append(s)
                if i + 1 < len(L) and L[i + 1].strip() != '': out.append(''); mod = True
                continue
            out.append(line)
        if mod: open(f, 'w').write('\n'.join(out)); changed.append(f)
    return changed

def punct(c): return c != '' and unicodedata.category(c)[0] in 'PS'
def ws(c): return c == '' or c.isspace()
def left_ok(p, n): return not ws(n) and (not punct(n) or ws(p) or punct(p))
def right_ok(p, n): return not ws(p) and (not punct(p) or ws(n) or punct(n))
MOVABLE = '。，；：！？、'

def fix_bold(files):
    tot = 0; skipped = []
    for f in files:
        lines = open(f).read().split('\n'); fence = None; nf = 0
        for i, line in enumerate(lines):
            body = re.sub(r'^[\s>]*', '', line); fm = re.match(r'(`{3,}|~{3,})', body)
            if fence:
                if fm and body.strip() == body.strip()[0] * len(body.strip()): fence = None
                continue
            if fm: fence = fm.group(1); continue
            if '**' not in line: continue
            # mask code spans and math so their contents are never touched or counted
            mask = re.sub(r'(`+)(?!`).+?(?<!`)\1(?!`)', lambda m: '\x01' * len(m.group(0)), line)
            mask = re.sub(r'(?<!\\)\$\$.+?\$\$|(?<![\\$])\$(?![\s$])(?:\\.|[^$\\])+?(?<!\s)\$(?!\$)', lambda m: '\x01' * len(m.group(0)), mask)
            pos = [m.start() for m in re.finditer(r'(?<!\*)\*\*(?!\*)', mask)]
            if len(pos) % 2: skipped.append(f'{f}:{i+1}'); continue
            s = line; shift = 0
            for k in range(0, len(pos), 2):
                o, c = pos[k] + shift, pos[k + 1] + shift
                p, n = (s[o-1] if o else ''), s[o+2:o+3]
                if not left_ok(p, n):
                    s = s[:o] + ' ' + s[o:]; shift += 1; c += 1; nf += 1
                p2, n2 = s[c-1], s[c+2:c+3]
                if not right_ok(p2, n2):
                    if p2 in MOVABLE and not punct(s[c-2]):
                        s = s[:c-1] + '**' + p2 + s[c+2:]
                    else:
                        s = s[:c+2] + ' ' + s[c+2:]; shift += 1
                    nf += 1
            lines[i] = s
        if nf: open(f, 'w').write('\n'.join(lines)); tot += nf
    return tot, skipped

if __name__ == '__main__':
    dry = '--dry' in sys.argv
    files = [a for a in sys.argv[1:] if a != '--dry'] or [x for x in subprocess.run(['git', '-c', 'core.quotepath=off', 'ls-files', '-z', '*.md'], capture_output=True, text=True).stdout.split('\0') if x]
    before = {f: open(f).read() for f in files}
    total = collections.Counter()
    for f in files:
        st = collections.Counter(); new = fix_file(f, st); total.update(st)
        if new != before[f]: open(f, 'w').write(new)
    env = fix_display_env(files)
    nb, skipped = fix_bold(files)
    changed = [f for f in files if open(f).read() != before[f]]
    if dry:
        for f in changed: open(f, 'w').write(before[f])
    print(('[dry] ' if dry else '') + f'files changed: {len(changed)}/{len(files)}')
    if total: print('  math:', dict(total))
    if env: print('  display-env paragraphs:', env)
    if nb: print('  bold repairs:', nb)
    for x in flags: print('  FLAG', x)
    for x in skipped: print('  SKIP (odd ** count)', x)
