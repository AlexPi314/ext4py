"""Измерение покрытия строк пакета стандартными unittest и trace.

Сбор не ограничивается каталогами; в отчёт отбираются полные пути
основного пакета. Дочерние процессы в счётчики не входят.
"""

import argparse
import json
from pathlib import Path
import sys
import trace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def measure(output: Path) -> int:
    """Выполнить unittest под trace и сохранить покрытие основного пакета.

    Args:
        output: Каталог для построчных файлов .cover и summary.json.

    Returns:
        0 при успешном наборе тестов, иначе 1.

    Сводка учитывает только текущие модули src/ext4py. Файлы без
    зарегистрированных попаданий всё равно входят в знаменатель.
    """
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'tests'))
    # Фильтруем итоговые полные пути: ignoredirs может смешать одноимённые модули.
    tracer = trace.Trace(count=True, trace=False)

    def run():
        """Обнаружить тесты и вернуть TestResult их выполнения в текущем процессе."""
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        return unittest.TextTestRunner(verbosity=1).run(suite)

    result = tracer.runfunc(run)
    core = ROOT / 'src' / 'ext4py'
    counts = {key: value for key, value in tracer.results().counts.items()
              if Path(key[0]).parent == core}
    for path in core.glob('*.py'):
        if not any(filename == str(path) for filename, _ in counts):
            counts[str(path), 0] = 0
    output.mkdir(parents=True, exist_ok=True)
    trace.CoverageResults(counts=counts).write_results(show_missing=True, summary=True, coverdir=str(output))
    modules = {}
    expected = {f'ext4py.{path.stem}' for path in core.glob('*.py')}
    for path in output.glob('*.cover'):
        if path.stem not in expected:
            continue
        hit = missed = 0
        for line in path.read_text().splitlines():
            prefix = line[:7]
            if prefix.startswith('>>>>>>'):
                missed += 1
            elif prefix.strip().rstrip(':').isdigit():
                hit += 1
        modules[path.stem] = {'hit': hit, 'missed': missed, 'total': hit + missed,
                              'percent': round(100 * hit / (hit + missed), 2) if hit + missed else 100.0}
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'modules': modules}
    total = sum(item['total'] for item in modules.values())
    hit = sum(item['hit'] for item in modules.values())
    summary['total'] = {'hit': hit, 'total': total, 'percent': round(100 * hit / total, 2) if total else 100.0}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Покрытие строк: {hit}/{total} ({summary["total"]["percent"]}%)')
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    """Разобрать --output и вернуть код выполнения измерителя покрытия."""
    parser = argparse.ArgumentParser(description='Покрытие строк пакета ext4py стандартным модулем trace')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/coverage')
    args = parser.parse_args()
    return measure(args.output.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
