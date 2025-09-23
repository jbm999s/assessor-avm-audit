<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>

  <h1>Assessor AVM Audit</h1>
  <p><strong>Auditing the Cook County Assessor’s Automated Valuation Model (AVM) for transparency, accountability, and fairness</strong></p>

  <h2>Project Overview</h2>
  <p>This project re-implements and audits the Cook County Assessor’s Automated Valuation Model (AVM). The Assessor’s office uses machine learning (<strong>LightGBM decision trees</strong>) to generate property valuations. While the model is technically advanced, it is a black box for most taxpayers and practitioners.</p>

  <div class="highlight">
    <p>I adapted the public AVM code to create an audit tool that:</p>
    <ul>
      <li>Scores and ranks comparable properties based on similarity and model leaf signatures.</li>
      <li>Produces transparent CSV outputs showing how the Assessor’s chosen comps rank against all potential comps.</li>
      <li>Allows filtering by timeframe (e.g., 3 years of sales) and adjusting weights between similarity and leaf match.</li>
      <li>Provides reproducible results for appeals, hearings, and audit purposes.</li>
    </ul>
    <p>The result is a tool that exposes bias or cherry-picking in comp selection while following appraisal standards of transparency and documentation.</p>
  </div>

  <h2>Why It Matters</h2>
  <p>Property assessments drive tax bills. If comps are cherry-picked or applied inconsistently, taxpayers may be over-assessed and over-taxed.</p>
  <p>This tool matters because it:</p>
  <ul>
    <li>Audits the Assessor with their own rules: same AVM, same data, but transparent.</li>
    <li>Supports appeals: shows better comps that were overlooked.</li>
    <li>Aligns with <strong>USPAP principles</strong>:
      <ul>
        <li><strong>Sales Comparison Approach (Standards 1 & 2)</strong> – uses recent sales of similar properties.</li>
        <li><strong>Scope of Work Rule</strong> – clearly defines the dataset and methodology.</li>
        <li><strong>Record Keeping Rule</strong> – produces a full record of comps, scores, and ranks for reproducibility.</li>
      </ul>
    </li>
  </ul>

  <h2>How It Works</h2>
  <h3>Inputs</h3>
  <ul>
    <li><strong>Assessment PIN parquet file</strong> – base property data.</li>
    <li><strong>Leaf signatures parquet file</strong> – model outputs from LightGBM.</li>
    <li><strong>Optional Assessor comps CSV</strong> – the official “five comps” for side-by-side comparison.</li>
  </ul>

  <h3>Process</h3>
  <ol>
    <li>Builds a candidate pool of all properties within the selected scope (e.g., township).</li>
    <li>Computes two scores:
      <ul>
        <li>Similarity score (feature closeness).</li>
        <li>Leaf score (model path overlap).</li>
      </ul>
    </li>
    <li>Combines them into a composite score (default 70% similarity, 30% leaf).</li>
    <li>Ranks all properties by composite score.</li>
  </ol>

  <h3>Outputs</h3>
  <ul>
    <li><code>Top-K comps CSV</code> (e.g., best 30 matches).</li>
    <li><code>Assessor compare CSV</code> (their comps ranked against the full universe).</li>
  </ul>

  <div class="highlight">
    <p><strong>Example:</strong></p>
    <ul>
      <li><code>PIN_comps_w70-30.csv</code> → my Top-K comps.</li>
      <li><code>PIN_ccao_compare_w70-30.csv</code> → how the Assessor’s comps rank, with building AV, land AV, FMV, sqft metrics, and flags for whether they made Top-K.</li>
    </ul>
  </div>

  <h2>Getting Started</h2>
  <h3>Setup</h3>
  <pre><code>git clone https://github.com/yourusername/assessor-avm-audit.git
cd assessor-avm-audit
pip install -r requirements.txt</code></pre>

  <h3>Quick Run</h3>
  <pre><code>python3 scripts/get_comps.py 05214180020000 \
  --k 30 \
  --data output/assessment_pin/model_assessment_pin.parquet \
  --leaves output/intermediate/pin_leaves.parquet \
  --scope township \
  --outdir output/comp_sheets \
  --ccao data/ccao_5.csv --ccao_pin_col PIN</code></pre>

  <h3>Configuration</h3>
  <ul>
    <li><code>--k</code> → number of comps to return (default: 30).</li>
    <li><code>--scope</code> → geographic boundary (township, neighborhood).</li>
    <li><code>--weight</code> → similarity/leaf ratio (default: 70/30).</li>
    <li><code>--sales_window</code> → filter by sale date, e.g. 3 years.</li>
    <li><code>--outdir</code> → output directory for CSVs.</li>
  </ul>

  <h2>Tech Stack</h2>
  <ul>
    <li>Python 3</li>
    <li>Pandas / NumPy – data wrangling.</li>
    <li>LightGBM – model structure and leaf signatures.</li>
    <li>Parquet – efficient data storage.</li>
  </ul>

  <h2>Attribution</h2>
  <p>This project builds on the Cook County Assessor’s Office open-source AVM.</p>
  <p><strong>Their contribution:</strong> model training, feature engineering, LightGBM outputs.</p>
  <p><strong>My contribution:</strong> audit layer for scoring, weighting, Top-K comp selection, Assessor compare reports, and appeal-ready CSV outputs.</p>
  <p>###</p>
  <p><a href="http://www.justinmcclelland.com" target="_new">Justin McClelland</a></p>

</body>
</html>
