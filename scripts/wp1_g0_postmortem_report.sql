-- 本文件由 wp1_g0_postmortem_report.py 实际执行，用于生成报告部件的最终快照。
-- 上游诊断变换见 scripts/wp1_g0_postmortem.py；这里保持最终选择、排序与呈现粒度可审计。

-- dataset: headline
SELECT * FROM headline;

-- dataset: gate_metrics
SELECT * FROM gate_metrics ORDER BY "order";

-- dataset: class_metrics
SELECT * FROM class_metrics ORDER BY delta ASC;

-- dataset: session_rows
SELECT * FROM session_rows ORDER BY cohort, session;

-- dataset: distribution_summary
SELECT * FROM distribution_summary ORDER BY cohort;

-- dataset: low_energy
SELECT * FROM low_energy ORDER BY threshold;

-- dataset: qa_checks
SELECT * FROM qa_checks ORDER BY "check";

-- dataset: history
SELECT * FROM history;

-- dataset: chart_map
SELECT * FROM chart_map ORDER BY section;
