import React from "react";
import { createRoot } from "react-dom/client";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./style.css";

type Outcome = { status: string; correct: boolean | null };
type Fixture = {
  id: string;
  suite: string;
  question: string;
  document: string;
  expected_answer: string;
  fact_spans: string[];
  fact_position: string | null;
  target_context_length: number | null;
  actual_token_length: number | null;
  answer_type: string;
};
type Result = {
  fixture_id: string;
  config_id: string;
  status: string;
  answer: string;
  error: string | null;
  outcome: Outcome | null;
  memory_rss_bytes: number | null;
  peak_process_rss_bytes: number | null;
  prompt_tokens: number[];
  timings: Record<string, unknown>;
  end_to_end_ms: number | null;
};
type Pair = {
  fixture_id: string;
  context: number | null;
  config_id: string;
  classification: string;
  output_disagreement: boolean;
  baseline_answer: string | null;
  candidate_answer: string | null;
  baseline_status: string | null;
  candidate_status: string | null;
  candidate_runtime_status: string | null;
  candidate_error: string | null;
};
type Config = { id: string; cache_type_k: string; cache_type_v: string };
type Exp = {
  id: string;
  created_at: string;
  status: string;
  model_path: string;
  model_sha256: string;
  runtime_sha256: string;
  runtime_version: string;
  hardware: Record<string, string>;
  settings: Record<string, any>;
  fixtures: Fixture[];
  configurations: Config[];
  results: Result[];
  pairings: Pair[];
  unsupported: { config_id?: string; reason: string; status?: string }[];
};
type Tab = "overview" | "fidelity" | "failures" | "context";
const fmt = (x: number | null | undefined, d = 1) =>
  x == null ? "unavailable" : `${x.toFixed(d)}%`;

function App() {
  const [experiments, setExperiments] = React.useState<
      { id: string; created_at: string; status: string }[]
    >([]),
    [exp, setExp] = React.useState<Exp | null>(null),
    [tab, setTab] = React.useState<Tab>("overview");
  const [configFilter, setConfigFilter] = React.useState("all"),
    [selectedConfigIds, setSelectedConfigIds] = React.useState<string[]>([]),
    [workload, setWorkload] = React.useState("all"),
    [context, setContext] = React.useState("all"),
    [pairFilter, setPairFilter] = React.useState("all"),
    [metric, setMetric] = React.useState("accuracy");
  React.useEffect(() => {
    fetch("/api/experiments")
      .then((r) => r.json())
      .then(setExperiments)
      .catch(() => setExperiments([]));
  }, []);
  async function open(id: string) {
    const result = await fetch(`/api/experiments/${id}`).then((r) => r.json());
    setExp(result);
    setTab("overview");
    setConfigFilter("all");
    setSelectedConfigIds([]);
    setWorkload("all");
    setContext("all");
  }
  const fixtures = React.useMemo(
    () => new Map((exp?.fixtures || []).map((f) => [f.id, f])),
    [exp],
  );
  const relevant = (exp?.results || []).filter((r) => {
    const f = fixtures.get(r.fixture_id);
    return (
      (configFilter === "all" || r.config_id === configFilter) &&
      (workload === "all" || f?.suite === workload) &&
      (context === "all" || String(f?.target_context_length) === context)
    );
  });
  const accuracy = (configId: string, ctx?: number, suite?: string) => {
    const rows = (exp?.results || []).filter((r) => {
      const f = fixtures.get(r.fixture_id);
      return (
        r.config_id === configId &&
        r.status === "complete" &&
        r.outcome &&
        r.outcome.correct !== null &&
        (ctx == null || f?.target_context_length === ctx) &&
        (suite == null || f?.suite === suite)
      );
    });
    return rows.length
      ? {
          value:
            (rows.filter((r) => r.outcome?.correct).length / rows.length) * 100,
          n: rows.length,
        }
      : null;
  };
  const baselineId = exp?.settings.baseline_config_id || "baseline";
  const visibleConfigs = (exp?.configurations || []).filter(
    (c) => selectedConfigIds.length === 0 || selectedConfigIds.includes(c.id),
  );
  function toggleConfig(id: string, checked: boolean) {
    const all = (exp?.configurations || []).map((c) => c.id);
    const current = selectedConfigIds.length ? selectedConfigIds : all;
    const next = checked ? [...current, id] : current.filter((x) => x !== id);
    setSelectedConfigIds(next.length === all.length ? [] : next);
  }
  const summary = (exp?.configurations || []).map((c) => {
    const rows = (exp?.results || []).filter((r) => r.config_id === c.id),
      valid = rows.filter(
        (r) =>
          r.status === "complete" && r.outcome && r.outcome.correct !== null,
      ),
      good = valid.filter((r) => r.outcome?.correct).length;
    const pairs = (exp?.pairings || []).filter((p) => p.config_id === c.id);
    const retained = pairs.filter(
        (p) =>
          p.baseline_status === "correct" && p.candidate_status === "correct",
      ).length,
      baseCorrect = pairs.filter((p) => p.baseline_status === "correct").length;
    const rss = rows
      .map((r) => r.peak_process_rss_bytes)
      .filter((x): x is number => x !== null);
    const latency = rows
      .map((r) => r.end_to_end_ms)
      .filter((x): x is number => x !== null);
    const kv = Object.entries(
      exp?.settings.kv_allocation_by_config_context || {},
    )
      .filter(([k, v]: [string, any]) => k.endsWith(`:${c.id}`) && v.available)
      .map(([k, v]: [string, any]) => ({
        context: Number(k.split(":")[0]),
        bytes: v.total_bytes as number,
      }));
    return {
      c,
      good,
      n: valid.length,
      acc: valid.length ? (good / valid.length) * 100 : null,
      ret:
        c.id === baselineId
          ? valid.length
            ? 100
            : null
          : baseCorrect
            ? (retained / baseCorrect) * 100
            : null,
      reg: pairs.filter((p) => p.classification === "regression").length,
      improve: pairs.filter((p) => p.classification === "improvement").length,
      rss: rss.length ? Math.max(...rss) : null,
      ms: latency.length
        ? latency.reduce((a, b) => a + b, 0) / latency.length
        : null,
      kv,
    };
  });
  const contexts = [
    ...new Set(
      (exp?.fixtures || [])
        .map((f) => f.target_context_length)
        .filter((v): v is number => v !== null),
    ),
  ].sort((a, b) => a - b);
  const selectedPairs = (exp?.pairings || []).filter((p) => {
    const f = fixtures.get(p.fixture_id);
    const matchesOutcome =
      pairFilter === "all" ||
      (pairFilter === "disagreement"
        ? p.output_disagreement
        : pairFilter === "unanswerable_failures"
          ? f?.answer_type === "unanswerable" &&
            p.candidate_status !== "abstained_correctly"
          : p.classification === pairFilter);
    return (
      matchesOutcome &&
      (configFilter === "all" || p.config_id === configFilter) &&
      (workload === "all" || f?.suite === workload) &&
      (context === "all" || String(f?.target_context_length) === context)
    );
  });
  const accuracyRows = contexts.flatMap((ctx) =>
    (exp?.configurations || []).map((c) => {
      const a = accuracy(c.id, ctx, workload === "all" ? undefined : workload);
      return {
        context: ctx,
        config: c.id,
        value: a?.value ?? null,
        n: a?.n ?? 0,
      };
    }),
  );
  const contextRows = contexts.flatMap((ctx) =>
    (exp?.configurations || []).map((c) => {
      const rows = (exp?.results || []).filter((r) => {
        const f = fixtures.get(r.fixture_id);
        return (
          r.config_id === c.id &&
          f?.target_context_length === ctx &&
          (workload === "all" || f?.suite === workload)
        );
      });
      let value: number | null = null,
        n = 0;
      if (metric === "accuracy") {
        const valid = rows.filter(
          (r) =>
            r.status === "complete" && r.outcome && r.outcome.correct !== null,
        );
        n = valid.length;
        value = n
          ? (valid.filter((r) => r.outcome?.correct).length / n) * 100
          : null;
      } else if (metric === "retention") {
        if (c.id === baselineId) {
          const base = accuracy(
            c.id,
            ctx,
            workload === "all" ? undefined : workload,
          );
          n = base?.n ?? 0;
          value = n ? 100 : null;
        } else {
          const paired = (exp?.pairings || []).filter(
            (p) =>
              p.config_id === c.id &&
              p.context === ctx &&
              (workload === "all" ||
                fixtures.get(p.fixture_id)?.suite === workload),
          );
          const baselineOk = paired.filter(
            (p) => p.baseline_status === "correct",
          );
          n = baselineOk.length;
          value = n
            ? (baselineOk.filter((p) => p.candidate_status === "correct")
                .length /
                n) *
              100
            : null;
        }
      } else if (metric === "hallucination") {
        const unanswerable = rows.filter(
          (r) =>
            fixtures.get(r.fixture_id)?.answer_type === "unanswerable" &&
            r.outcome,
        );
        n = unanswerable.length;
        value = n
          ? (unanswerable.filter(
              (r) => r.outcome?.status === "unsupported_assertion",
            ).length /
              n) *
            100
          : null;
      } else if (metric === "speed") {
        const speeds = rows
          .map((r) => r.timings.prompt_tokens_per_second)
          .filter((x): x is number => typeof x === "number");
        n = speeds.length;
        value = n ? speeds.reduce((a, b) => a + b, 0) / n : null;
      } else if (metric === "rss") {
        const rss = rows
          .map((r) => r.peak_process_rss_bytes)
          .filter((x): x is number => x !== null);
        n = rss.length;
        value = n ? Math.max(...rss) / 1024 ** 3 : null;
      } else if (metric === "kv") {
        const allocation =
          exp?.settings.kv_allocation_by_config_context?.[`${ctx}:${c.id}`];
        value = allocation?.available
          ? allocation.total_bytes / 1024 ** 3
          : null;
        n = value === null ? 0 : 1;
      }
      return { context: ctx, config: c.id, value, n };
    }),
  );
  const kvPoints = summary.flatMap((x) =>
    contexts
      .filter((c) => context === "all" || String(c) === context)
      .map((c) => {
        const allocation =
          exp?.settings.kv_allocation_by_config_context?.[`${c}:${x.c.id}`];
        let a = accuracy(x.c.id, c, workload === "all" ? undefined : workload);
        if (metric === "retention") {
          if (x.c.id === baselineId) {
            a = a ? { value: 100, n: a.n } : null;
          } else {
            const paired = (exp?.pairings || []).filter(
              (p) =>
                p.config_id === x.c.id &&
                p.context === c &&
                (workload === "all" ||
                  fixtures.get(p.fixture_id)?.suite === workload),
            );
            const baselineOk = paired.filter(
              (p) => p.baseline_status === "correct",
            );
            a = baselineOk.length
              ? {
                  value:
                    (paired.filter(
                      (p) =>
                        p.baseline_status === "correct" &&
                        p.candidate_status === "correct",
                    ).length /
                      baselineOk.length) *
                    100,
                  n: baselineOk.length,
                }
              : null;
          }
        }
        return allocation?.available && a
          ? {
              name: x.c.id,
              config: x.c.id,
              context: c,
              kvGB: allocation.total_bytes / 1024 ** 3,
              value: a.value,
              n: a.n,
            }
          : null;
      })
      .filter((x): x is NonNullable<typeof x> => x !== null),
  );
  const visibleKvPoints = kvPoints.filter((p) =>
    visibleConfigs.some((c) => c.id === p.config),
  );
  function Highlight({ fixture }: { fixture: Fixture }) {
    let parts: React.ReactNode[] = [fixture.document];
    for (const span of fixture.fact_spans || []) {
      parts = parts.flatMap((p) => {
        if (typeof p !== "string" || !span) return [p];
        const i = p.indexOf(span);
        return i < 0
          ? [p]
          : [
              p.slice(0, i),
              <mark key={`${span}-${i}`}>{span}</mark>,
              p.slice(i + span.length),
            ];
      });
    }
    return <pre>{parts}</pre>;
  }
  function ConfigurationPicker() {
    return (
      <fieldset className="config-picker">
        <legend>Configurations</legend>
        {(exp?.configurations || []).map((c) => (
          <label key={c.id}>
            <input
              type="checkbox"
              checked={
                selectedConfigIds.length === 0 ||
                selectedConfigIds.includes(c.id)
              }
              onChange={(e) => toggleConfig(c.id, e.target.checked)}
            />
            {c.id}
          </label>
        ))}
      </fieldset>
    );
  }
  return (
    <main>
      <header>
        <div className="mark">CF</div>
        <div>
          <p className="eyebrow">LOCAL INFERENCE RESEARCH</p>
          <h1>KV Cache Quality Benchmark</h1>
        </div>
        <span className="local">● LOCAL DATA</span>
      </header>
      <section className="intro">
        <div>
          <p className="eyebrow">BEHAVIORAL FIDELITY, MEASURED</p>
          <h2>What does cache compression cost your model?</h2>
          <p>
            Paired correctness, fact retention, memory, and timing across local
            llama.cpp cache configurations.
          </p>
        </div>
        <aside>
          <strong>{experiments.length}</strong>
          <span>saved experiments</span>
        </aside>
      </section>
      <div className="layout">
        <nav>
          <h3>Experiments</h3>
          {experiments.length ? (
            experiments.map((e) => (
              <button
                className={exp?.id === e.id ? "active" : ""}
                key={e.id}
                onClick={() => open(e.id)}
              >
                <b>{new Date(e.created_at).toLocaleString()}</b>
                <small>
                  {e.id.slice(0, 8)} · {e.status}
                </small>
              </button>
            ))
          ) : (
            <p className="muted">
              No saved experiments. Run{" "}
              <code>kv-cache-quality-benchmark run</code> to add one.
            </p>
          )}
        </nav>
        <article>
          {exp ? (
            <>
              <div className="title-row">
                <div>
                  <p className="eyebrow">EXPERIMENT {exp.id.slice(0, 8)}</p>
                  <h2>{exp.status.toUpperCase()}</h2>
                </div>
                <span className="badge">
                  {exp.results.length} trials · {exp.fixtures.length} fixtures
                </span>
              </div>
              <div className="meta">
                <span>
                  Model <b>{exp.model_path.split("/").pop()}</b>
                </span>
                <span>
                  Model SHA-256{" "}
                  <b title={exp.model_sha256}>
                    {exp.model_sha256.slice(0, 16)}…
                  </b>
                </span>
                <span>
                  llama.cpp <b>{exp.runtime_version}</b>
                </span>
                <span>
                  Hardware <b>{exp.hardware.machine}</b>
                </span>
                <span>
                  Backend{" "}
                  <b>
                    {exp.settings.backend_requested ||
                      exp.settings.runtime_backend_build_mentions?.join(", ") ||
                      "runtime default"}
                  </b>
                </span>
              </div>
              <div className="tabs" role="tablist">
                {(["overview", "fidelity", "failures", "context"] as Tab[]).map(
                  (t) => (
                    <button
                      role="tab"
                      aria-selected={tab === t}
                      className={tab === t ? "chosen" : ""}
                      key={t}
                      onClick={() => setTab(t)}
                    >
                      {t === "fidelity"
                        ? "Fidelity vs memory"
                        : t === "failures"
                          ? "Failure explorer"
                          : t === "context"
                            ? "Context degradation"
                            : "Overview"}
                    </button>
                  ),
                )}
              </div>
              {tab === "overview" && (
                <>
                  <section className="panel">
                    <h3>Configuration comparison</h3>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Config</th>
                            <th>Accuracy</th>
                            <th>Retention</th>
                            <th>Regressions</th>
                            <th>Improvements</th>
                            <th>KV allocation</th>
                            <th>Peak RSS</th>
                            <th>Mean latency</th>
                          </tr>
                        </thead>
                        <tbody>
                          {summary.map((x) => (
                            <tr key={x.c.id}>
                              <td>
                                {x.c.id}
                                <small className="sub">
                                  K {x.c.cache_type_k} / V {x.c.cache_type_v}
                                </small>
                              </td>
                              <td>
                                {x.n
                                  ? `${x.good}/${x.n} · ${fmt(x.acc)}`
                                  : "unavailable"}
                              </td>
                              <td>
                                {x.ret === null ? "unavailable" : fmt(x.ret)}
                              </td>
                              <td>{x.reg}</td>
                              <td>{x.improve}</td>
                              <td>
                                {x.kv.length
                                  ? x.kv.map((v: any) => (
                                      <span key={v.context} className="sub">
                                        {v.context}:{" "}
                                        {(v.bytes / 1024 ** 3).toFixed(3)} GiB
                                      </span>
                                    ))
                                  : "unavailable"}
                              </td>
                              <td>
                                {x.rss
                                  ? `${(x.rss / 1024 ** 3).toFixed(2)} GiB`
                                  : "unavailable"}
                              </td>
                              <td>
                                {x.ms ? `${x.ms.toFixed(1)} ms` : "unavailable"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="note">
                      Accuracy denominator excludes failed and invalid answers.
                      RSS is sampled process memory, not KV allocation. KV
                      allocation is unavailable unless runtime logs explicitly
                      reported it.
                    </p>
                  </section>
                  <section className="panel">
                    <h3>Accuracy across contexts</h3>
                    <p className="note">
                      Each point is measured accuracy for the selected stored
                      fixture workload.
                    </p>
                    <div
                      className="chart"
                      role="img"
                      aria-label="Answer accuracy by context length for each selected cache configuration"
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={accuracyRows}>
                          <CartesianGrid stroke="#27323a" />
                          <XAxis
                            dataKey="context"
                            type="number"
                            domain={["dataMin", "dataMax"]}
                            tickFormatter={(x) => `${x / 1024}K`}
                            stroke="#92a0a8"
                          />
                          <YAxis domain={[0, 100]} stroke="#92a0a8" />
                          <Tooltip
                            formatter={(v: any, _n: any, p: any) => [
                              `${Number(v).toFixed(1)}% (n=${p.payload.n})`,
                              p.payload.config,
                            ]}
                          />
                          {visibleConfigs.map((c, i) => (
                            <Line
                              key={c.id}
                              data={accuracyRows.filter(
                                (x) => x.config === c.id,
                              )}
                              dataKey="value"
                              name={c.id}
                              type="monotone"
                              connectNulls={false}
                              stroke={
                                ["#a8e66b", "#57c6c2", "#e9a45b", "#ac91e9"][
                                  i % 4
                                ]
                              }
                              strokeWidth={2}
                              dot
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </section>
                </>
              )}
              {tab === "fidelity" && (
                <>
                  <div className="filters">
                    <label>
                      Context
                      <select
                        value={context}
                        onChange={(e) => setContext(e.target.value)}
                      >
                        <option value="all">All</option>
                        {contexts.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Workload
                      <select
                        value={workload}
                        onChange={(e) => setWorkload(e.target.value)}
                      >
                        <option value="all">All workloads</option>
                        {[
                          "needle",
                          "retention",
                          "distractor",
                          "multi_fact",
                          "unanswerable",
                        ].map((x) => (
                          <option key={x}>{x}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Y metric
                      <select
                        value={metric}
                        onChange={(e) => setMetric(e.target.value)}
                      >
                        <option value="accuracy">Accuracy</option>
                        <option value="retention">
                          Baseline-conditioned retention
                        </option>
                      </select>
                    </label>
                    <ConfigurationPicker />
                  </div>
                  {visibleKvPoints.length > 0 ? (
                    <section className="panel">
                      <h3>
                        Measured KV allocation vs{" "}
                        {metric === "accuracy"
                          ? "accuracy"
                          : "baseline-conditioned retention"}
                      </h3>
                      <p className="note">
                        X is explicit allocation from runtime logs, Y is{" "}
                        {metric}. Tooltip shows the configuration, context, and
                        denominator.
                      </p>
                      <div
                        className="chart"
                        role="img"
                        aria-label="Measured KV allocation against answer fidelity"
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart>
                            <CartesianGrid stroke="#27323a" />
                            <XAxis
                              type="number"
                              dataKey="kvGB"
                              name="Measured KV allocation"
                              unit=" GiB"
                              stroke="#92a0a8"
                            />
                            <YAxis
                              type="number"
                              dataKey="value"
                              name={metric}
                              domain={[0, 100]}
                              unit="%"
                              stroke="#92a0a8"
                            />
                            <Tooltip
                              cursor={{ strokeDasharray: "3 3" }}
                              formatter={(v: any, n: any, p: any) => [
                                `${Number(v).toFixed(1)}${n === "value" ? "%" : " GiB"} · n=${p.payload.n}`,
                                n,
                              ]}
                              labelFormatter={(_: any, p: any) =>
                                `${p?.[0]?.payload.config}, context ${p?.[0]?.payload.context}`
                              }
                            />
                            {visibleConfigs.map((c, i) => (
                              <Scatter
                                key={c.id}
                                name={c.id}
                                data={visibleKvPoints.filter(
                                  (p) => p.config === c.id,
                                )}
                                fill={
                                  c.id === baselineId
                                    ? "#a8e66b"
                                    : ["#57c6c2", "#e9a45b", "#ac91e9"][i % 3]
                                }
                                shape={c.id === baselineId ? "star" : "circle"}
                              />
                            ))}
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="note">
                        Memory provenance: explicit llama.cpp allocation log
                        lines. The F16 reference uses a star marker.
                      </p>
                    </section>
                  ) : (
                    <section className="panel">
                      <h3>
                        Measured KV allocation vs{" "}
                        {metric === "accuracy"
                          ? "accuracy"
                          : "baseline-conditioned retention"}
                      </h3>
                      <div className="empty compact">
                        <h3>KV allocation unavailable</h3>
                        <p>
                          The runtime did not emit a recognized allocation line
                          for the selected configurations. No RSS or theoretical
                          estimate is substituted on this axis.
                        </p>
                      </div>
                    </section>
                  )}
                  <section className="panel">
                    <h3>Peak process RSS by configuration</h3>
                    <p className="note">
                      RSS is a sampled process measurement and includes
                      model/runtime memory, not just KV storage.
                    </p>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Configuration</th>
                            <th>Peak sampled RSS</th>
                            <th>Context filter</th>
                          </tr>
                        </thead>
                        <tbody>
                          {summary
                            .filter((x) =>
                              visibleConfigs.some((c) => c.id === x.c.id),
                            )
                            .map((x) => (
                              <tr key={x.c.id}>
                                <td>{x.c.id}</td>
                                <td>
                                  {x.rss
                                    ? `${(x.rss / 1024 ** 3).toFixed(2)} GiB`
                                    : "unavailable"}
                                </td>
                                <td>{context}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              )}
              {tab === "failures" && (
                <>
                  <div className="filters">
                    <label>
                      Outcome
                      <select
                        value={pairFilter}
                        onChange={(e) => setPairFilter(e.target.value)}
                      >
                        <option value="all">All comparisons</option>
                        <option value="regression">
                          Baseline correct, candidate wrong
                        </option>
                        <option value="improvement">
                          Baseline wrong, candidate correct
                        </option>
                        <option value="preserved_success">Both correct</option>
                        <option value="persistent_failure">Both wrong</option>
                        <option value="disagreement">Different outputs</option>
                        <option value="unanswerable_failures">
                          Unanswerable failures
                        </option>
                        <option value="comparison_unavailable">
                          Unavailable pairing
                        </option>
                      </select>
                    </label>
                    <label>
                      Configuration
                      <select
                        value={configFilter}
                        onChange={(e) => setConfigFilter(e.target.value)}
                      >
                        <option value="all">All configurations</option>
                        {exp.configurations
                          .filter((c) => c.id !== baselineId)
                          .map((c) => (
                            <option key={c.id}>{c.id}</option>
                          ))}
                      </select>
                    </label>
                    <label>
                      Workload
                      <select
                        value={workload}
                        onChange={(e) => setWorkload(e.target.value)}
                      >
                        <option value="all">All workloads</option>
                        {[
                          "needle",
                          "retention",
                          "distractor",
                          "multi_fact",
                          "unanswerable",
                        ].map((x) => (
                          <option key={x}>{x}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Context
                      <select
                        value={context}
                        onChange={(e) => setContext(e.target.value)}
                      >
                        <option value="all">All contexts</option>
                        {contexts.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <section className="panel">
                    <h3>Paired test outcomes</h3>
                    <p className="note">
                      Output disagreement is separate from correctness
                      classification. Highlighted spans are fixture ground
                      truth; they do not represent model attention.
                    </p>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Test</th>
                            <th>Workload</th>
                            <th>Context</th>
                            <th>Fact position</th>
                            <th>Expected</th>
                            <th>F16 answer</th>
                            <th>Optimized answer</th>
                            <th>Outcome</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedPairs.map((p, i) => {
                            const f = fixtures.get(p.fixture_id);
                            const r = exp.results.find(
                              (x) =>
                                x.fixture_id === p.fixture_id &&
                                x.config_id === p.config_id,
                            );
                            return (
                              <tr key={`${p.fixture_id}-${p.config_id}-${i}`}>
                                <td>
                                  <details>
                                    <summary>{p.fixture_id}</summary>
                                    {f && (
                                      <>
                                        <p>
                                          <b>Question</b>
                                          <br />
                                          {f.question}
                                        </p>
                                        <p>
                                          <b>
                                            Document with fixture facts
                                            highlighted
                                          </b>
                                          <Highlight fixture={f} />
                                        </p>
                                        <p>
                                          <b>Expected</b> {f.expected_answer}
                                          <br />
                                          <b>Baseline</b>{" "}
                                          {p.baseline_answer || "unavailable"}
                                          <br />
                                          <b>Candidate</b>{" "}
                                          {p.candidate_answer || "unavailable"}
                                        </p>
                                        <p>
                                          Prompt tokens:{" "}
                                          {r?.prompt_tokens.length ??
                                            "unavailable"}{" "}
                                          · score:{" "}
                                          {r?.outcome?.status ??
                                            r?.status ??
                                            "not recorded"}
                                        </p>
                                        {(r?.error || p.candidate_error) && (
                                          <p>
                                            Error:{" "}
                                            {r?.error || p.candidate_error}
                                          </p>
                                        )}
                                      </>
                                    )}
                                  </details>
                                </td>
                                <td>{f?.suite}</td>
                                <td>{p.context}</td>
                                <td>{f?.fact_position || "—"}</td>
                                <td>{f?.expected_answer}</td>
                                <td>{p.baseline_answer || "—"}</td>
                                <td>{p.candidate_answer || "—"}</td>
                                <td>
                                  {p.classification}
                                  {p.output_disagreement
                                    ? " · disagreement"
                                    : ""}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              )}
              {tab === "context" && (
                <>
                  <div className="filters">
                    <label>
                      Workload
                      <select
                        value={workload}
                        onChange={(e) => setWorkload(e.target.value)}
                      >
                        <option value="all">All workloads</option>
                        {[
                          "needle",
                          "retention",
                          "distractor",
                          "multi_fact",
                          "unanswerable",
                        ].map((x) => (
                          <option key={x}>{x}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Metric
                      <select
                        value={metric}
                        onChange={(e) => setMetric(e.target.value)}
                      >
                        <option value="accuracy">Accuracy</option>
                        <option value="retention">
                          Baseline-conditioned retention
                        </option>
                        <option value="hallucination">
                          Unsupported assertions
                        </option>
                        <option value="speed">Prompt processing speed</option>
                        <option value="kv">KV allocation</option>
                        <option value="rss">Peak RSS</option>
                      </select>
                    </label>
                    <ConfigurationPicker />
                  </div>
                  <section className="panel">
                    <h3>Context length comparison · {metric}</h3>
                    <p className="note">
                      Configured context is shown separately from actual
                      prompt-token count. Missing metrics remain unavailable.
                      Workload: {workload}.
                    </p>
                    {contextRows.some((x) => x.value !== null) ? (
                      <div
                        className="chart"
                        role="img"
                        aria-label={`${metric} by configured context length`}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={contextRows}>
                            <CartesianGrid stroke="#27323a" />
                            <XAxis
                              dataKey="context"
                              type="number"
                              domain={["dataMin", "dataMax"]}
                              tickFormatter={(x) => `${x / 1024}K`}
                              stroke="#92a0a8"
                            />
                            <YAxis
                              domain={
                                [
                                  "accuracy",
                                  "retention",
                                  "hallucination",
                                ].includes(metric)
                                  ? [0, 100]
                                  : ["auto", "auto"]
                              }
                              stroke="#92a0a8"
                            />
                            <Tooltip
                              formatter={(v: any, _n: any, p: any) => [
                                `${Number(v).toFixed(2)}${["accuracy", "retention", "hallucination"].includes(metric) ? "%" : metric === "speed" ? " tokens/s" : " GiB"} (n=${p.payload.n})`,
                                p.payload.config,
                              ]}
                            />
                            {visibleConfigs.map((c, i) => (
                              <Line
                                key={c.id}
                                data={contextRows.filter(
                                  (x) => x.config === c.id,
                                )}
                                dataKey="value"
                                name={c.id}
                                stroke={
                                  ["#a8e66b", "#57c6c2", "#e9a45b", "#ac91e9"][
                                    i % 4
                                  ]
                                }
                                connectNulls={false}
                                dot
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <p className="empty compact">
                        This metric was not emitted or recognized in the stored
                        measurements.
                      </p>
                    )}
                  </section>
                  <section className="panel">
                    <h3>Fact-position accuracy</h3>
                    <p className="note">
                      Fixture positions are measured in token space from the
                      serialized prompt. Cells show correct / scored
                      denominator.
                    </p>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Configuration</th>
                            {["Beginning", "25%", "Middle", "75%", "End"].map(
                              (x) => (
                                <th key={x}>{x}</th>
                              ),
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {visibleConfigs.map((c) => (
                            <tr key={c.id}>
                              <th>{c.id}</th>
                              {["Beginning", "25%", "Middle", "75%", "End"].map(
                                (bucket) => {
                                  const rows = relevant.filter(
                                    (r) =>
                                      r.config_id === c.id &&
                                      fixtures.get(r.fixture_id)?.suite ===
                                        "needle" &&
                                      r.outcome &&
                                      r.outcome.correct !== null &&
                                      (() => {
                                        const n = parseFloat(
                                          fixtures.get(r.fixture_id)
                                            ?.fact_position || "0",
                                        );
                                        return (
                                          [
                                            "Beginning",
                                            "25%",
                                            "Middle",
                                            "75%",
                                            "End",
                                          ][
                                            Math.min(
                                              4,
                                              Math.max(0, Math.round(n / 25)),
                                            )
                                          ] === bucket
                                        );
                                      })(),
                                  );
                                  const correct = rows.filter(
                                    (r) => r.outcome?.correct,
                                  ).length;
                                  return (
                                    <td
                                      key={bucket}
                                      className={
                                        rows.length && correct === rows.length
                                          ? "good"
                                          : rows.length
                                            ? "bad"
                                            : ""
                                      }
                                    >
                                      {rows.length
                                        ? `${correct}/${rows.length}`
                                        : "—"}
                                    </td>
                                  );
                                },
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              )}
              {exp.unsupported.length > 0 && (
                <section className="panel">
                  <h3>Unsupported or skipped</h3>
                  {exp.unsupported.map((u, i) => (
                    <p key={i}>
                      {u.config_id} · {u.status || "skipped"}: {u.reason}
                    </p>
                  ))}
                </section>
              )}
            </>
          ) : (
            <div className="empty">
              <div className="empty-icon">↗</div>
              <h2>Select a saved experiment</h2>
              <p>
                The dashboard reads persisted local results and does not create
                placeholder measurements.
              </p>
            </div>
          )}
        </article>
      </div>
      <footer>
        KV Cache Quality Benchmark · Fixture-defined ground truth · Small
        samples are exploratory
      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
