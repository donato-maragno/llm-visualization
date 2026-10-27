/* Measured, saved forward passes. Interactive after completion; never a live animation. */
"use strict";
window.gpt2View = (() => {
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const tokenText = value => String(value).replace(/ /g,"␠").replace(/\n/g,"↵").replace(/\t/g,"⇥");
  const stages = [["embeddings","01","Text → embeddings"],["attention","02","Self-attention"],["compute","03","Transformer computations"],["probabilities","04","Next-token probabilities"]];
  let run=null, runId=null, question=0, variant=0, layer=0, head=0, query=0, key=0, stage="attention", matrixMode="attention", lastFrame="", hasSnapshot=false, onSelection=null;
  const num = n => Number(n).toFixed(4);
  const pct = n => `${(100*n).toFixed(2)}%`;
  const signedColor = (value, scale=1) => {
    const t=Math.min(1,Math.abs(value)/(scale||1));
    const target=value<0?[143,112,214]:[40,157,145];
    return `rgb(${[244,246,250].map((v,i)=>Math.round(v+(target[i]-v)*t)).join(",")})`;
  };
  const probabilityColor = value => `rgba(38,145,139,${.06+.94*Math.sqrt(Math.max(0,Math.min(1,value)))})`;
  const name = v => v.technique==="layers"?`Bypass last ${v.strength} layer(s)`:v.technique==="heads"?`Mask ${v.strength} head(s) / layer`:`MLP ${v.technique} · ${v.strength}%`;
  const root = () => $("gpt2-view");
  const isWalkthrough=()=>root().dataset.mode==="walkthrough";
  const heading = status => `<div class="panel-heading"><div><div class="eyebrow">${isWalkthrough()?"GPT-2 / FIXED SENTENCE WALKTHROUGH":"TESTED QUESTION / TRANSFORMER PIPELINE"}</div><h2>${isWalkthrough()?"One sentence. See inside the transformer.":"Same question. Two model states."}</h2></div><span id="viz-status" class="tag">${status}</span></div>`;
  const position = (p,i) => p.tokens[i].position??i;
  function vector(values, scale) {
    scale=scale||Math.max(...values.map(Math.abs),.000001);
    return `<div class="vector-cells">${values.map((v,i)=>`<svg viewBox="0 0 40 25" role="img" aria-label="Dimension ${i}: ${num(v)}"><title>Dimension ${i}: ${num(v)}</title><rect width="40" height="25" rx="3" fill="${signedColor(v,scale)}"/><text x="20" y="15" text-anchor="middle">${num(v)}</text></svg>`).join("")}</div>`;
  }
  function embeddingPanel(p, scales) {
    const fields=p.position_encoding==="rotary"?[["token","Learned token embedding"]]:[["token","Learned token embedding"],["position","Learned position embedding"],["combined","Token + position"]];
    const note=p.position_encoding==="rotary"?'<p class="caption">Qwen uses rotary positions (RoPE) inside Q/K attention, not a learned position vector added to the input. The computation panel shows Q/K after RoPE.</p>':"";
    return note+fields.map(([kind,title])=>`<section class="tensor-section"><h4>${title}</h4>${p.tokens.map((t,i)=>`<div class="vector-row${i===query?" selected":""}"><span title="Prompt position ${position(p,i)}, ID ${t.id}">${position(p,i)} · ${esc(tokenText(t.text))}</span>${vector(p.embeddings[kind][i],scales[kind])}</div>`).join("")}</section>`).join("");
  }
  function attentionMatrix(p, data, scale) {
    const matrix=data[matrixMode==="attention"?"attention":"scores"][head], n=p.tokens.length, cell=18, pad=24, size=pad+n*cell;
    const labels=p.tokens.map((t,i)=>`<text x="${pad+i*cell+cell/2}" y="14" text-anchor="middle">${position(p,i)}</text><text x="14" y="${pad+i*cell+12}" text-anchor="middle">${position(p,i)}</text>`).join("");
    const cells=matrix.flatMap((row,i)=>row.map((v,j)=>{
      const masked=j>i&&matrixMode!=="scores", selected=i===query&&j===key;
      const fill=masked?"#e5e8ee":matrixMode==="attention"?probabilityColor(v):signedColor(v,scale);
      const detail=masked?"Causal mask: future token unavailable":matrixMode==="attention"?`Attention ${pct(v)}`:`Scaled Q·K score ${num(v)}`;
      return `<g data-cell="${i},${j}" role="button" tabindex="0" aria-label="Query token ${i}, key token ${j}: ${esc(detail)}"><rect x="${pad+j*cell}" y="${pad+i*cell}" width="${cell-1}" height="${cell-1}" rx="2" fill="${fill}" class="${selected?"matrix-selected":i===query?"matrix-query":""}"/><title>${esc(tokenText(p.tokens[i].text))} → ${esc(tokenText(p.tokens[j].text))}: ${esc(detail)}</title></g>`;
    })).join("");
    return `<svg class="attention-matrix" viewBox="0 0 ${size} ${size}" role="group" aria-label="${esc(matrixMode)} matrix. Rows are queries; columns are keys.">${labels}${cells}</svg>`;
  }
  function connections(p, data) {
    const weights=data.attention[head][query];
    const strongest=weights.map((value,index)=>({value,index})).filter(x=>x.index<=query).sort((a,b)=>b.value-a.value).slice(0,6);
    const height=30+strongest.length*30;
    return `<div class="connection-map"><p class="caption">Strongest links among displayed tokens for prompt position ${position(p,query)} (up to six)</p><svg viewBox="0 0 520 ${height}" role="img" aria-label="Attention derived from measured Q/K">${strongest.map(({value,index},i)=>{
      const y=24+i*30;
      return `<path d="M120,${height/2} C215,${height/2} 215,${y} 292,${y}" fill="none" stroke="${probabilityColor(value)}" stroke-width="${1+value*9}"/><text x="300" y="${y+4}">${position(p,index)} · ${esc(tokenText(p.tokens[index].text))} · ${pct(value)}</text>`;
    }).join("")}<rect x="4" y="${height/2-14}" width="116" height="28" rx="7" fill="#e4f3ef"/><text x="12" y="${height/2+4}">${position(p,query)} · ${esc(tokenText(p.tokens[query].text))}</text></svg></div>`;
  }
  function maskNote(data) {
    if(data.bypassed)return '<p class="mask-note bypass">Block bypassed. Internal attention and MLP are computed but discarded; the block output equals its input.</p>';
    if(data.masked_heads.includes(head))return '<p class="mask-note removed">This head is masked before the output projection. Its attention is still computed, but its output contribution is zero.</p>';
    if(data.masked_weights)return `<p class="mask-note removed">${data.masked_weights.toLocaleString()} / ${data.total_mlp_weights.toLocaleString()} MLP parameters masked in this block.</p>`;
    return '<p class="mask-note kept">This selected head and block remain connected.</p>';
  }
  function attentionPanel(p, data, scale) {
    const omitted=data.omitted_attention?`<p class="caption">Attention to unshown prompt positions: <b>${pct(data.omitted_attention[head][query])}</b>. Normalized over the full causal context, not just this displayed sample.</p>`:"";
    return `${maskNote(data)}<p class="caption">Rows: query tokens · columns: key tokens. Labels are positions in the actual ${isWalkthrough()?"fixed sentence":"question input"}. Gray cells cannot access future tokens. Select a cell to inspect that pair.</p>${attentionMatrix(p,data,scale)}<p class="matrix-reading">Position <b>${position(p,query)}</b> → position <b>${position(p,key)}</b>: ${key>query&&matrixMode!=="scores"?"causally blocked":matrixMode==="attention"?pct(data.attention[head][query][key]):num(data.scores[head][query][key])}</p>${omitted}${connections(p,data)}`;
  }
  function computePanel(p, data, scales) {
    const sample=(title,values,scale)=>`<section class="tensor-section"><h4>${title}</h4>${vector(values,scale)}</section>`;
    const positional=p.position_encoding==="rotary"?" → rotary positions on Q/K (RoPE)":"";
    const activation=p.model_type==="qwen2"?"gated SiLU MLP":"MLP (linear → GELU → linear)";
    return `${maskNote(data)}<div class="qkv-formula">Layer norm → learned Q, K, V projections${positional}</div>${sample(`Query · position ${position(p,query)}`,data.q[head][query],scales.q)}${sample(`Key · position ${position(p,key)}`,data.k[head][key],scales.k)}${sample(`Value · position ${position(p,key)}`,data.v[head][key],scales.v)}<div class="calculation"><span>Q · K / √${p.head_dim}</span><strong>${num(data.scores[head][query][key])}</strong><span>${key>query?"Causal mask → −∞ → softmax = 0":`Causal mask → softmax = ${pct(data.attention[head][query][key])}`}</span></div>${sample("Σ attention × V · actual head contribution",data.head_output[head][query],scales.head_output)}<div class="qkv-formula">Concatenate heads → projection + residual → layer norm → ${activation} + residual</div>${sample("MLP output · selected token",data.mlp_output[query],scales.mlp_output)}${sample("Residual stream · block input",data.input[query],scales.input)}${sample("Residual stream · effective block output",data.output[query],scales.output)}<p class="caption">Vectors show the first ${p.vector_sample_size} dimensions, not the whole vector. Q·K uses all ${p.head_dim} head dimensions. Attention is reconstructed from measured Q/K over the full context; head, MLP and effective block outputs are captured from the real forward pass.</p>`;
  }
  function probabilitiesPanel(p) {
    const prediction=p.predictions[query];
    return `<p class="caption">After prompt position ${position(p,query)}: <b>${esc(tokenText(p.tokens[query].text))}</b>. Final layer norm → language-model projection → full-vocabulary softmax (temperature 1, before decoding controls).</p><ol class="probability-list">${prediction.candidates.map((candidate,i)=>`<li><span class="candidate-rank">${i+1}</span><code title="Token ID ${candidate.id}">${esc(tokenText(candidate.text))}</code><svg class="probability-track" viewBox="0 0 100 14" preserveAspectRatio="none" role="img" aria-label="${pct(candidate.probability)}"><rect class="probability-background" width="100" height="14" rx="3"/><rect class="probability-bar" width="${candidate.probability*100}" height="14" rx="3"/></svg><strong>${pct(candidate.probability)}</strong></li>`).join("")}</ol><p class="caption">Other vocabulary tokens: <b>${pct(prediction.other_probability)}</b>. Top candidates are not renormalized to 100%.</p>`;
  }
  function continuation(answer, prompt) {
    const stop=answer.stop_reason==="context_limit"||answer.truncated?"context limit (text may be unfinished)":answer.stop_reason==="eos"?"end-of-text token":answer.stop_reason||"finished";
    return `<details class="continuation" open><summary>Generated continuation · ${answer.tokens} tokens · ${esc(stop)}</summary><div><span class="prompt-prefix">${esc(prompt)}</span><span>${esc(answer.text)}</span></div></details>`;
  }
  function paintLegacy() {
    const baseline=run.baselines?.[0], choices=Object.keys(run.comparisons||{}).map(k=>+k.split(":")[0]);
    if(!choices.includes(variant))variant=choices[0]??0;
    const pruned=run.comparisons?.[`${variant}:0`]?.answer;
    if(!baseline?.pipeline||!pruned?.pipeline){root().innerHTML=heading("NO PIPELINE SNAPSHOT")+'<p class="muted compact">This record has no paired forward-pass data. Enter a text prompt and start a new GPT-2 comparison. Older Python-coding results are preserved, but cannot be converted into activation measurements.</p>';return;}
    const a=baseline.pipeline,b=pruned.pipeline;
    if(!hasSnapshot){query=a.tokens.length-1;key=query;hasSnapshot=true;}
    layer=Math.min(layer,a.layers.length-1);head=Math.min(head,a.head_count-1);query=Math.min(query,a.tokens.length-1);key=Math.min(key,a.tokens.length-1);
    const la=a.layers[layer],lb=b.layers[layer];
    const maxAbs=values=>Math.max(...values.flat(Infinity).map(Math.abs),.000001);
    const scales={};
    for(const field of ["q","k","v","head_output","mlp_output","input","output"])scales[field]=Math.max(maxAbs(la[field]),maxAbs(lb[field]));
    for(const field of ["token","position","combined"])scales[field]=Math.max(maxAbs(a.embeddings[field]),maxAbs(b.embeddings[field]));
    const scoreScale=Math.max(maxAbs(la.scores),maxAbs(lb.scores));
    const panel=(p,data)=>stage==="embeddings"?embeddingPanel(p,scales):stage==="attention"?attentionPanel(p,data,scoreScale):stage==="compute"?computePanel(p,data,scales):probabilitiesPanel(p);
    root().innerHTML=`${heading(run.status==="completed"?"FINAL · SAVED COMPARISON":"PARTIAL · SAVED COMPARISON")}<p id="viz-context" class="muted compact">A frozen forward pass on the same prompt for both model states. Explore it dynamically below; no live generation animation. Original weights are restored after every variant.</p><div class="pipeline-controls"><label>Pruned model<select id="pipeline-variant">${choices.map(i=>`<option value="${i}"${variant===i?" selected":""}>${esc(name(run.config.variants[i]))}</option>`).join("")}</select></label><label>Transformer layer<select id="pipeline-layer">${a.layers.map((_,i)=>`<option value="${i}"${layer===i?" selected":""}>Layer ${i+1}${b.layers[i].bypassed?" · bypassed":""}</option>`).join("")}</select></label><label>Attention head<select id="pipeline-head">${Array.from({length:a.head_count},(_,i)=>`<option value="${i}"${head===i?" selected":""}>Head ${i+1}${lb.masked_heads.includes(i)?" · masked":""}</option>`).join("")}</select></label></div><div class="pipeline-steps" aria-label="Pipeline stages">${stages.map(([id,n,title])=>`<button type="button" data-stage="${id}" aria-pressed="${stage===id}"><small>${n}</small>${title}<span>→</span></button>`).join("")}</div><div class="token-picker"><div class="eyebrow">SELECT A QUERY TOKEN · INDEX / TOKEN / ID IN TOOLTIP</div><div>${a.tokens.map((t,i)=>`<button type="button" data-token="${i}" aria-pressed="${query===i}" title="Position ${i} · token ID ${t.id}"><small>${i}</small>${esc(tokenText(t.text))}</button>`).join("")}</div></div>${stage==="attention"?`<div class="matrix-modes" aria-label="Attention computation step">${[["scores","QKᵀ / √d"],["masked","Causal mask"],["attention","Softmax attention"]].map(([id,title])=>`<button type="button" data-matrix="${id}" aria-pressed="${matrixMode===id}">${title}</button>`).join("")}</div>`:""}<div class="pipeline-legend"><span>Teal = positive / stronger probability</span><span>Violet = negative vector value</span><span>Gray = causal mask</span><span>Shared color scales across both models</span></div><div class="pipeline-pair"><article class="pipeline-lane original"><header><span class="eyebrow">ORIGINAL GPT-2</span><strong>All pathways kept</strong></header>${panel(a,la)}</article><article class="pipeline-lane pruned"><header><span class="eyebrow">PRUNED GPT-2</span><strong>${esc(name(run.config.variants[variant]))}</strong></header>${panel(b,lb)}</article></div><div class="pipeline-pair generated-pair"><article>${continuation(baseline,run.config.prompt)}</article><article>${continuation(pruned,run.config.prompt)}</article></div><p class="caption">Only the input prompt is inspected, not the growing continuation. Token selection predicts the token after that input position. Continuations use greedy decoding, repetition penalty 1.1 and no repeated 3-grams; predictions above show raw probabilities. Generation ends at EOS, cancellation or context exhaustion. Masking does not remove dense parameters or guarantee speed gains. Interface concept inspired by <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener noreferrer">Transformer Explainer</a>; all displayed measurements come from this local model.</p>`;
  }
  function questionControls(a, b) {
    return `<div class="pipeline-controls"><label class="pipeline-question-control">Tested question<select id="pipeline-question">${run.config.problems.map((problem,i)=>`<option value="${i}"${question===i?" selected":""}>${i+1}. ${esc(problem.title)} · ${problem.tests.length} tests</option>`).join("")}</select></label><label>Pruning variant<select id="pipeline-variant">${run.config.variants.map((v,i)=>`<option value="${i}"${variant===i?" selected":""}>${esc(name(v))}</option>`).join("")}</select></label>${a&&b?`<label>Transformer layer<select id="pipeline-layer">${a.layers.map((_,i)=>`<option value="${i}"${layer===i?" selected":""}>Layer ${i+1}${b.layers[i].bypassed?" · bypassed":""}</option>`).join("")}</select></label><label>Attention head<select id="pipeline-head">${Array.from({length:a.head_count},(_,i)=>`<option value="${i}"${head===i?" selected":""}>Head ${i+1}${b.layers[layer].masked_heads.includes(i)?" · masked":""}</option>`).join("")}</select></label>`:""}</div>`;
  }
  function questionContext(problem, baseline, pruned, cell) {
    const output=(answer,i)=>answer?.execution?.cases?.[i]?JSON.stringify(answer.execution.cases[i].output):answer?.execution?.status||"Not run";
    const correctness=answer=>answer?.execution?`${answer.execution.passed}/${answer.execution.total}`:"pending";
    return `<section class="question-context"><div class="eyebrow">QUESTION ${question+1} · ${esc(problem.id)} · SAME TESTS FOR BOTH MODELS</div><h3>${esc(problem.title)}</h3><p>${esc(problem.description)}</p><div class="question-outcomes"><span>Same tested outcomes: <b>${cell?.score??"—"}</b></span><span>Expected tests passed · original <b>${correctness(baseline)}</b> · pruned <b>${correctness(pruned)}</b></span></div><p class="caption">${esc(cell?.reason||"The visualization follows this question's solution generation. Actual output comparison is performed by the Python runner.")}</p><details class="pipeline-test-details"><summary>${problem.tests.length} actual test cases and outputs for this question</summary><div class="pipeline-test-scroll"><table><thead><tr><th>Input arguments</th><th>Expected</th><th>Original output</th><th>Pruned output</th></tr></thead><tbody>${problem.tests.map((test,i)=>`<tr><td><code>${esc(JSON.stringify(test.args))}</code></td><td><code>${esc(JSON.stringify(test.expected))}</code></td><td><code>${esc(output(baseline,i))}</code></td><td><code>${esc(output(pruned,i))}</code></td></tr>`).join("")}</tbody></table></div></details></section>`;
  }
  function solutionPanel(answer) {
    const execution=answer?.execution;
    return `<details class="continuation"><summary>Generated Python solution · ${answer?.tokens??0} tokens · ${esc(execution?.status||"not run")}</summary><div>${esc(execution?.code??answer?.text??"Not generated")}</div>${execution?.error?`<p class="caption">${esc(execution.error)}</p>`:""}</details>`;
  }
  function paintWalkthrough() {
    const baseline=run.baselines?.[0],pruned=run.comparisons?.[`${variant}:0`]?.answer;
    const preset=run.walkthrough_presets[variant],a=baseline?.pipeline,b=pruned?.pipeline;
    const title=heading(run.status==="completed"?"FINAL · SAVED WALKTHROUGH":"PARTIAL · SAVED WALKTHROUGH");
    const mechanism=`<section class="walkthrough-mechanism"><div class="eyebrow">PRUNING EXAMPLE ${variant+1} / ${esc(preset.technique)}</div><h3>${esc(preset.title)}</h3><p>${esc(preset.explanation)}</p><p><b>What to inspect:</b> ${esc(preset.inspect)}</p></section>`;
    if(!a||!b){root().innerHTML=`${title}${mechanism}<p class="muted compact">${esc(baseline?.pipeline_error||pruned?.pipeline_error||(baseline?.pipeline_available&&pruned?.pipeline_available?"Loading this preset's measured model states…":"This example is preparing automatically. Its measured states will appear when ready."))}</p>`;return;}
    if(a.prompt!==run.config.prompt||b.prompt!==run.config.prompt||a.tokens.map(t=>t.id).join()!==b.tokens.map(t=>t.id).join()){
      root().innerHTML=`${title}<p class="mask-note removed">These model states do not use the same fixed sentence; they are not displayed.</p>`;return;
    }
    if(!hasSnapshot){query=a.tokens.length-1;key=query;hasSnapshot=true;}
    layer=Math.min(layer,a.layers.length-1);head=Math.min(head,a.head_count-1);query=Math.min(query,a.tokens.length-1);key=Math.min(key,a.tokens.length-1);
    const la=a.layers[layer],lb=b.layers[layer];
    const maxAbs=values=>Math.max(...values.flat(Infinity).map(Math.abs),.000001),scales={};
    for(const field of ["q","k","v","head_output","mlp_output","input","output"])scales[field]=Math.max(maxAbs(la[field]),maxAbs(lb[field]));
    for(const field of ["token","position","combined"])scales[field]=Math.max(maxAbs(a.embeddings[field]),maxAbs(b.embeddings[field]));
    const scoreScale=Math.max(maxAbs(la.scores),maxAbs(lb.scores));
    const panel=(p,data)=>stage==="embeddings"?embeddingPanel(p,scales):stage==="attention"?attentionPanel(p,data,scoreScale):stage==="compute"?computePanel(p,data,scales):probabilitiesPanel(p);
    const originalTop=a.predictions[a.tokens.length-1].candidates[0],prunedTop=b.predictions[b.tokens.length-1].candidates[0];
    const lesson={embeddings:"The sentence is split into tokens. Learned token vectors and position vectors are added. These presets do not prune embeddings, so the two inputs are identical.",attention:"Each query token can attend to earlier tokens and itself, never future tokens. Follow the colored links, then compare scores, causal masking and softmax. Masking a head removes its contribution, not its internal attention calculation.",compute:"Q, K and V are learned transformations of the input. Attention mixes values; projection, residual connections and the MLP update the representation. Compare measured vectors to see which contribution was actually removed.",probabilities:"The final representation becomes logits for every vocabulary token. Softmax makes a probability distribution. These are raw next-token probabilities—not math correctness or proof that a continuation makes sense."}[stage];
    root().innerHTML=`${title}
      ${mechanism}
      <div class="walkthrough-effect"><span>Original preferred next token: <code>${esc(tokenText(originalTop.text))}</code> ${pct(originalTop.probability)}</span><span>Pruned preferred next token: <code>${esc(tokenText(prunedTop.text))}</code> ${pct(prunedTop.probability)}</span><strong>${originalTop.id===prunedTop.id?"Top token unchanged; probabilities can still change.":"The preferred next token changed."}</strong></div>
      <div class="pipeline-pair walkthrough-outputs"><article><div class="walkthrough-output-title">Original GPT-2 · generated text</div>${continuation(baseline,run.config.prompt)}</article><article><div class="walkthrough-output-title">Pruned GPT-2 · ${esc(preset.title)}</div>${continuation(pruned,run.config.prompt)}</article></div>
      <div class="pipeline-controls"><label>Predefined pruning example<select id="pipeline-variant">${run.walkthrough_presets.map((p,i)=>`<option value="${i}"${variant===i?" selected":""}>${esc(p.title)}</option>`).join("")}</select></label><label>Transformer layer<select id="pipeline-layer">${a.layers.map((_,i)=>`<option value="${i}"${layer===i?" selected":""}>Layer ${i+1}${b.layers[i].bypassed?" · bypassed":""}</option>`).join("")}</select></label><label>Attention head<select id="pipeline-head">${Array.from({length:a.head_count},(_,i)=>`<option value="${i}"${head===i?" selected":""}>Head ${i+1}${lb.masked_heads.includes(i)?" · masked":""}</option>`).join("")}</select></label></div>
      <div class="eyebrow">THE TRANSFORMER STACK / SELECT A LAYER</div><div class="walkthrough-layer-rail">${b.layers.map((l,i)=>`<button type="button" data-layer="${i}" class="${l.bypassed?"bypassed":l.masked_heads.length||l.masked_weights?"masked":""}" aria-pressed="${layer===i}" title="${l.bypassed?"Block output bypassed":l.masked_heads.length?`${l.masked_heads.length} heads masked`:l.masked_weights?`${l.masked_weights} MLP parameters masked`:"All pathways kept"}">L${i+1}</button>`).join("")}</div>
      <div class="pipeline-steps" aria-label="Pipeline stages">${stages.map(([id,n,label])=>`<button type="button" data-stage="${id}" aria-pressed="${stage===id}"><small>${n}</small>${label}<span>→</span></button>`).join("")}</div><p class="walkthrough-step-copy">${lesson}</p>
      <div class="token-picker"><div class="eyebrow">INSPECT A TOKEN / THIS DOES NOT CHANGE THE SENTENCE</div><div>${a.tokens.map((t,i)=>`<button type="button" data-token="${i}" aria-pressed="${query===i}" title="Position ${i} · token ID ${t.id}"><small>${i}</small>${esc(tokenText(t.text))}</button>`).join("")}</div></div>
      ${stage==="attention"?`<div class="matrix-modes">${[["scores","QKᵀ / √d"],["masked","Causal mask"],["attention","Softmax attention"]].map(([id,label])=>`<button type="button" data-matrix="${id}" aria-pressed="${matrixMode===id}">${label}</button>`).join("")}</div>`:""}
      <div class="pipeline-legend"><span>Teal: positive / stronger attention</span><span>Violet: negative vector value</span><span>Gray: future tokens blocked</span><span>Layer rail: coral = mask · amber = bypass</span><span>Shared color scales</span></div>
      <div class="pipeline-pair"><article class="pipeline-lane original"><header><span class="eyebrow">ORIGINAL GPT-2</span><strong>All pathways kept</strong></header>${panel(a,la)}</article><article class="pipeline-lane pruned"><header><span class="eyebrow">PRUNED GPT-2</span><strong>${esc(preset.title)}</strong></header>${panel(b,lb)}</article></div>
      <p class="caption">These are saved measurements for the fixed input sentence, not animations of every generated token. The last input token predicts the first continuation token; earlier token selections inspect earlier prefixes. Vector tiles show the first eight dimensions. Prediction bars use the full-vocabulary softmax without renormalizing the top eight. Text uses greedy decoding, repetition penalty 1.1 and no repeated 3-grams, so raw preferred tokens can differ from the final decoding choice. Masking does not remove dense parameters or guarantee speed or memory savings. Inspired by <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener noreferrer">Transformer Explainer</a>.</p>`;
  }
  function paint() {
    if(run.kind==="walkthrough"){paintWalkthrough();return;}
    if(!run.config.problems){
      paintLegacy();
      root().insertAdjacentHTML("afterbegin",'<p class="mask-note bypass">Legacy standalone text run—not a tested-question comparison. New runs use the question bank and actual solution tests.</p>');
      return;
    }
    const problem=run.config.problems[question],baseline=run.baselines?.[question],cell=run.comparisons?.[`${variant}:${question}`],pruned=cell?.answer;
    const a=baseline?.pipeline,b=pruned?.pipeline;
    const context=questionContext(problem,baseline,pruned,cell);
    const title=heading(run.status==="completed"?"FINAL · QUESTION COMPARISON":"PARTIAL · QUESTION COMPARISON");
    if(!a||!b){
      const loading=baseline?.pipeline_available&&pruned?.pipeline_available&&!baseline?.pipeline_error&&!pruned?.pipeline_error;
      root().innerHTML=`${title}${questionControls()}${context}<p class="muted compact">${loading?"Loading measured pipelines for this tested question…":esc(baseline?.pipeline_error||pruned?.pipeline_error||"This question has no paired pipeline snapshot. Its solution and test results are still shown. Run a new question comparison to record measurements.")}</p>`;
      return;
    }
    // Never show another question's or model input's tensors under these results.
    if(a.problem_id!==problem.id||b.problem_id!==problem.id||a.prompt!==b.prompt||a.tokens.map(t=>t.id+":"+t.position).join()!==b.tokens.map(t=>t.id+":"+t.position).join()){
      root().innerHTML=`${title}${questionControls()}${context}<p class="mask-note removed">The saved pipelines do not match this question and input. They are not displayed.</p>`;return;
    }
    if(!hasSnapshot){query=a.tokens.length-1;key=query;hasSnapshot=true;}
    layer=Math.min(layer,a.layers.length-1);head=Math.min(head,a.head_count-1);query=Math.min(query,a.tokens.length-1);key=Math.min(key,a.tokens.length-1);
    const la=a.layers[layer],lb=b.layers[layer];
    const maxAbs=values=>Math.max(...values.flat(Infinity).map(Math.abs),.000001),scales={};
    for(const field of ["q","k","v","head_output","mlp_output","input","output"])scales[field]=Math.max(maxAbs(la[field]),maxAbs(lb[field]));
    for(const field of Object.keys(a.embeddings))scales[field]=Math.max(maxAbs(a.embeddings[field]),maxAbs(b.embeddings[field]));
    const scoreScale=Math.max(maxAbs(la.scores),maxAbs(lb.scores));
    const panel=(p,data)=>stage==="embeddings"?embeddingPanel(p,scales):stage==="attention"?attentionPanel(p,data,scoreScale):stage==="compute"?computePanel(p,data,scales):probabilitiesPanel(p);
    root().innerHTML=`${title}
      <p id="viz-context" class="muted compact">Transformer-style inspection of the exact prompt used for this tested question. Both sides use the same question input; the right side records the selected pruning mask. Final snapshots only, never live animation.</p>
      ${questionControls(a,b)}${context}
      <div class="pipeline-steps" aria-label="Pipeline stages">${stages.map(([id,n,label])=>`<button type="button" data-stage="${id}" aria-pressed="${stage===id}"><small>${n}</small>${label}<span>→</span></button>`).join("")}</div>
      <p class="caption">Showing ${a.tokens.length} ${a.sampled_positions?"sampled ":""}positions from this question's actual ${a.input_token_count}-token model input, including the final input token. The model processes the complete prepared prompt; display sampling does not change the question or tests.</p>
      ${a.prompt_truncated?`<p class="mask-note bypass">Generation's context handling truncated the original ${a.original_input_token_count}-token prompt to ${a.input_token_count} tokens. This snapshot shows the actual input used for generation, not the discarded positions.</p>`:""}
      <details class="pipeline-input-details"><summary>Exact model input for ${esc(problem.title)}</summary><pre class="pipeline-model-input">${esc(a.input_text??a.prompt)}</pre></details>
      <div class="token-picker"><div class="eyebrow">SELECT A QUERY TOKEN · ACTUAL PROMPT POSITIONS</div><div>${a.tokens.map((t,i)=>`<button type="button" data-token="${i}" aria-pressed="${query===i}" title="Prompt position ${position(a,i)} · token ID ${t.id}"><small>${position(a,i)}</small>${esc(tokenText(t.text))}</button>`).join("")}</div></div>
      ${stage==="attention"?`<div class="matrix-modes" aria-label="Attention computation step">${[["scores","QKᵀ / √d"],["masked","Causal mask"],["attention","Softmax attention"]].map(([id,label])=>`<button type="button" data-matrix="${id}" aria-pressed="${matrixMode===id}">${label}</button>`).join("")}</div>`:""}
      <div class="pipeline-legend"><span>Teal = positive / stronger probability</span><span>Violet = negative vector value</span><span>Gray = causal mask</span><span>Shared scales across both models</span></div>
      <div class="pipeline-pair"><article class="pipeline-lane original"><header><span class="eyebrow">ORIGINAL · ${esc(run.model)}</span><strong>${esc(problem.title)} · all pathways kept</strong></header>${panel(a,la)}</article><article class="pipeline-lane pruned"><header><span class="eyebrow">PRUNED · ${esc(run.model)}</span><strong>${esc(name(run.config.variants[variant]))}</strong></header>${panel(b,lb)}</article></div>
      <div class="pipeline-pair generated-pair"><article>${solutionPanel(baseline)}</article><article>${solutionPanel(pruned)}</article></div>
      <p class="caption">This is prompt processing for the selected question, not neural execution of each Python test case. Attention is reconstructed from measured Q/K over all causal keys; vector samples and output probabilities come from the real forward pass. The next-token chart predicts solution text, not whether a test passes. Full-vocabulary probabilities and hidden attention mass are not renormalized to the displayed subset. Test outcomes above come from the restricted Python runner. Interface inspired by <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener noreferrer">Transformer Explainer</a>.</p>`;
  }
  function render(record, visible, selectedQuestion=0, selectedVariant=0, selectionCallback=null) {
    root().hidden=!visible;if(!visible)return;
    const newRun=record?.id!==runId;
    if(newRun){runId=record?.id;layer=0;head=0;query=0;key=0;stage=isWalkthrough()?"embeddings":"attention";matrixMode="attention";lastFrame="";hasSnapshot=false;}
    if(record?.kind==="walkthrough"&&(newRun||variant!==selectedVariant)){
      const preset=record.walkthrough_presets[selectedVariant];layer=preset.inspection_layer;head=preset.inspection_head;
    }
    if(question!==selectedQuestion)hasSnapshot=false;
    question=selectedQuestion;variant=selectedVariant;onSelection=selectionCallback;
    run=record;
    const terminal=record&&!["running","queued","cancelling"].includes(record.status);
    const answer=record?.baselines?.[question],pruned=record?.comparisons?.[`${variant}:${question}`]?.answer;
    const frame=`${record?.id||"preview"}:${terminal?record.revision:"pending"}:${question}:${variant}:${!!answer?.pipeline}:${!!pruned?.pipeline}:${answer?.pipeline_error||""}`;
    if(frame===lastFrame)return;lastFrame=frame;
    if(!record){
      root().innerHTML=isWalkthrough()?`${heading("LOADING PREPARED EXAMPLE")}<p class="muted compact">The original and four pruned model states load automatically. Choose a preset to inspect its completed text and transformer computations.</p><div class="walkthrough-flow"><div><strong>01 · Input & embeddings</strong>The fixed sentence becomes tokens, numerical vectors and position information.</div><div><strong>02 · Self-attention</strong>Queries compare with keys to decide which earlier tokens matter.</div><div><strong>03 · Transformer blocks</strong>Attention outputs, residual connections and MLPs transform the token representations.</div><div><strong>04 · Output prediction</strong>Softmax ranks possible next tokens; repeated prediction completes the sentence.</div></div><p class="caption">Measured values are prepared once and cached, not invented for this preview.</p>`:heading("QUESTION-LINKED VISUALIZATION")+'<p class="muted compact">Select coding questions and pruning variants, then run the same tests on original and pruned solutions. After completion, choose a tested question to explore its embeddings, attention, Q/K/V, MLP and next-token predictions alongside its actual test outcomes.</p>';return;
    }
    if(!terminal){root().innerHTML=heading("COMPARISON IN PROGRESS")+'<p class="muted compact">Recording the original and pruned model states. Interactive embeddings, attention, Q/K/V and next-token probabilities appear after the run finishes. Generation progress is shown above; no live visualization is rendered.</p>';return;}
    paint();
  }
  document.addEventListener("DOMContentLoaded",()=>{
    const activate=event=>{
      const target=event.target.closest("[data-stage],[data-token],[data-matrix],[data-cell],[data-layer]");if(!target)return;
      if(event.type==="keydown"&&event.key!=="Enter"&&event.key!==" ")return;
      event.preventDefault();
      let selector;
      if(target.dataset.stage){stage=target.dataset.stage;selector=`[data-stage="${stage}"]`;}
      else if(target.dataset.token!==undefined){query=+target.dataset.token;key=Math.min(key,query);selector=`[data-token="${query}"]`;}
      else if(target.dataset.matrix){matrixMode=target.dataset.matrix;selector=`[data-matrix="${matrixMode}"]`;}
      else if(target.dataset.layer!==undefined){layer=+target.dataset.layer;selector=`[data-layer="${layer}"]`;}
      else { [query,key]=target.dataset.cell.split(",").map(Number);selector=`[data-cell="${query},${key}"]`; }
      paint();root().querySelector(selector)?.focus({preventScroll:true});
    };
    root().addEventListener("click",activate);root().addEventListener("keydown",activate);
    root().addEventListener("change",event=>{
      const id=event.target.id;
      if(id==="pipeline-question"){question=+event.target.value;hasSnapshot=false;}
      else if(id==="pipeline-variant"){
        variant=+event.target.value;
        if(run.kind==="walkthrough"){const preset=run.walkthrough_presets[variant];layer=preset.inspection_layer;head=preset.inspection_head;}
      }
      else if(id==="pipeline-layer")layer=+event.target.value;
      else if(id==="pipeline-head")head=+event.target.value;
      else return;
      if((id==="pipeline-question"||id==="pipeline-variant")&&onSelection){onSelection(question,variant);return;}
      paint();$(id)?.focus({preventScroll:true});
    });
  });
  return {render};
})();
