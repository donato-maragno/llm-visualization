"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const state = {token: "", busy: false, run: null, bank: [], revision: -1, active: null, pollTimer: null, selectedQ: 0, selectedV: 0};
const pipelineCache = new Map();
const techniques = {layers:"Layer bypass (N)", heads:"Head masking (N)", magnitude:"MLP magnitude (%)", random:"MLP random (%)"};
function parseProblem(source) {
  const p=JSON.parse(source);
  if(!p||typeof p!=="object"||Array.isArray(p)||typeof p.id!=="string"||typeof p.title!=="string"||typeof p.description!=="string"||!Array.isArray(p.tests)||!p.tests.length||p.tests.some(t=>!t||!Array.isArray(t.args)||!Object.hasOwn(t,"expected")))throw new Error("A problem needs id, title, description and tests with args arrays and expected values.");
  p.category=p.category||"arrays";p.normalize=p.normalize||"exact";
  if(!["arrays","strings","hashing","stack","dynamic-programming","search"].includes(p.category)||!["exact","unordered"].includes(p.normalize))throw new Error("Choose a supported category and exact/unordered normalization.");
  return p;
}
function notify(text) { $("notice").textContent = text; $("notice").hidden = !text; }
async function api(path, data) {
  const response = await fetch(path, {method:data === undefined ? "GET" : "POST", headers:{"Content-Type":"application/json", "X-Lab-Token":state.token}, ...(data === undefined ? {} : {body:JSON.stringify(data)})});
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail || `Request failed (${response.status})`));
  }
  return response.json();
}
function label(v) { return v.technique === "layers" ? `Bypass ${v.layer_position === "first" ? "first" : "last"} ${v.strength} layer(s)` : v.technique === "heads" ? `Mask ${v.strength}/12 heads` : `MLP ${v.technique} · ${v.strength}%`; }
function addVariant(technique="magnitude", strength=20) {
  if ($("variants").children.length >= 8) return notify("A run can compare up to eight variants.");
  const row = document.createElement("div"); row.className="variant-row";
  row.innerHTML = `<select aria-label="Pruning technique">${Object.entries(techniques).map(([value,text]) => `<option value="${value}"${value===technique?" selected":""}>${text}</option>`).join("")}</select><input type="number" aria-label="Pruning strength" value="${strength}" min="1" max="80" required><button type="button" class="remove" aria-label="Remove variant">×</button>`;
  const select=row.querySelector("select"), input=row.querySelector("input");
  const bounds=() => { input.max=select.value === "layers" ? ($("model").value==="gpt2"?11:27) : select.value === "heads" ? 11 : 80; if (+input.value > +input.max) input.value=input.max; estimate(); };
  select.addEventListener("change",bounds);input.addEventListener("input",estimate);
  row.querySelector("button").onclick=()=>{row.remove();estimate();};
  $("variants").append(row); bounds();
}
function getVariants() { return [...$("variants").children].map(row=>({technique:row.querySelector("select").value,strength:+row.querySelector("input").value})); }
function renderBank() {
  $("question-bank").innerHTML=state.bank.map((q,i)=>`<div class="question-row"><input type="checkbox" aria-label="Include problem ${i+1}" data-q="${i}"${q.selected?" checked":""}><span class="q-number">${String(i+1).padStart(2,"0")}</span><details class="problem-detail"><summary>${esc(q.problem.title)} <small>${q.problem.tests.length} tests · ${esc(q.problem.category)}</small></summary><p>${esc(q.problem.description)}</p><textarea rows="12" aria-label="Problem ${i+1} JSON" data-q="${i}">${esc(JSON.stringify(q.problem,null,2))}</textarea><span class="caption">Edit only if the solve(...) contract and expected test values remain consistent.</span></details><button type="button" class="remove" aria-label="Remove problem ${i+1}" data-remove="${i}">×</button></div>`).join("");
  $("question-bank").querySelectorAll("input,textarea").forEach(el=>{el.disabled=state.busy; el.addEventListener("input",()=>{const q=state.bank[+el.dataset.q];if(el.type==="checkbox")q.selected=el.checked;else{try{q.problem=parseProblem(el.value);q.error=null;el.setCustomValidity("");}catch(e){q.error=e.message;el.setCustomValidity(e.message);}}estimate();});});
  $("question-bank").querySelectorAll("button").forEach(el=>{el.disabled=state.busy;el.onclick=()=>{state.bank.splice(+el.dataset.remove,1);renderBank();};});
  estimate();
}
function estimate() {
  const n=state.bank.filter(q=>q.selected).length, v=$("variants").children.length;
  $("question-count").textContent=n; $("bank-summary").textContent=`Review ${state.bank.length} Python problems & tests · ${n} selected`;
  $("run-estimate").textContent=`${n} problems × ${v} variants · tested outcomes + final transformer comparison`;
}
function setBusy(value) {
  state.busy=value; $("settings").disabled=value;
  ["run","generate-question","add-question","custom-question","history","templates"].forEach(id=>$(id).disabled=value);
  $("cancel").hidden=!value; renderBank();
}
function loadDraft(run) {
  const config=run.config;
  $("model").value=config.model_name||run.model;
  $("seed").value=config.seed;
  if(config.problems){$("topic").value=config.topic;state.bank=config.problems.map(problem=>({problem,selected:true}));}
  $("variants").replaceChildren();config.variants.forEach(v=>addVariant(v.technique,v.strength));
  modelNote(false);renderBank();
}
function modelNote(preview=true) {
  $("model-footer").textContent="Qwen2.5-Coder · 1.5B · 32K context";
  $("task-intro").textContent="Compare original and pruned solutions on the same Python questions. Inspect their test outcomes and question-linked transformer pipelines.";
  $("footer-mode").textContent="PRUNING LAB / QUESTION-LINKED MODEL COMPARISONS";
  $("footer-metric").textContent="1 identical tested outcomes · 0 different outcomes / execution failure";
  $("model-note").textContent="Qwen-only coding experiments · 32,768 total context tokens · no user-set output cap. For the fixed GPT-2 sentence visualization, use the GPT-2 walkthrough menu tab.";
  $("run").innerHTML='Run comparison <span>→</span>';
  $("generation-note").textContent="No user-set generation token cap. Qwen has a 32,768-token context for prompt plus answer. Generation stops at EOS, cancellation, or native context exhaustion. Oversized inputs are reported, not silently shortened. Greedy decoding; seed controls random masks.";
  $("variants").querySelectorAll("select").forEach(select=>select.dispatchEvent(new Event("change")));
  if(preview)window.gpt2View.render(null,true);
}
function showView(name) {
  ["overview","answers"].forEach(view=>{$(view).hidden=view!==name;$("tab-"+view).setAttribute("aria-selected",view===name?"true":"false");});
}
function average(run, vi) {
  const cells=Object.entries(run.comparisons).filter(([k])=>k.startsWith(`${vi}:`)).map(([,v])=>v);
  const valid=cells.filter(c=>c.status==="completed" && Number.isFinite(c.score));
  return {value:valid.length?valid.reduce((sum,c)=>sum+c.score,0)/valid.length:null, n:valid.length, errors:cells.filter(c=>c.answer.execution.status!=="ok").length};
}
const formatted=n=>n===null||n===undefined?"—":String(n);
const tone=n=>n===null||n===undefined?"":n===1?"good":"bad";
function renderResults(run) {
  if(state.run?.id!==run.id)loadDraft(run);
  state.run=run;
  const textMode=run.evaluation==="gpt2-pipeline-v1";
  $("empty").hidden=true; $("results").hidden=textMode; $("progress-panel").hidden=false; $("export").disabled=false;
  $("run-status").textContent=run.status.toUpperCase(); $("stage").textContent=run.stage;
  $("progress").max=run.total; $("progress").value=run.done;
  $("progress-label").textContent=`${run.done} / ${run.total} ${textMode?"model states":"solutions"}${run.generated_tokens&&["baseline","pruned"].includes(run.phase)?` · ${run.generated_tokens} tokens`:""}`;
  if(textMode){renderSelectedPipeline();if(run.error)notify(run.error);return;}
  $("run-meta").textContent=`Python · ${run.model} · ${run.config.problems.length} problems · no user-set token cap (context-limited) · executed test outcomes · ${new Date(run.created_at*1000).toLocaleString()}`;
  $("summary-cards").innerHTML=run.config.variants.map((v,i)=>{const avg=average(run,i);return `<div class="metric"><div class="metric-title">${esc(label(v))}</div><strong>${avg.value===null?"—":`${(100*avg.value).toFixed(0)}%`}</strong><small>same outcomes · ${avg.n}/${run.config.problems.length} compared${avg.errors?` · ${avg.errors} execution failures`:""}</small></div>`;}).join("");
  $("matrix").innerHTML=`<thead><tr><th>PYTHON PROBLEM / SAME OUTCOME?</th>${run.config.variants.map(v=>`<th>${esc(label(v))}</th>`).join("")}</tr></thead><tbody>${run.config.problems.map((p,qi)=>`<tr><td><span class="index">${p.tests.length} TEST CASES</span>${esc(p.title)}</td>${run.config.variants.map((v,vi)=>{const c=run.comparisons[`${vi}:${qi}`];const working=["running","queued","cancelling"].includes(run.status);let text=working?(run.baselines[qi]?"Queued":"Original first"):"Not run";if(run.current_variant===vi&&run.current_question===qi)text=run.phase==="pruned"?"Generating…":run.phase==="testing"?"Testing…":text;if(c)text=formatted(c.score);return `<td><button type="button" class="score-button ${tone(c?.score)}" data-qi="${qi}" data-vi="${vi}" aria-label="Inspect problem ${qi+1}, ${esc(label(v))}">${text}</button></td>`;}).join("")}</tr>`).join("")}</tbody>`;
  $("matrix").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.selectedQ=+button.dataset.qi;state.selectedV=+button.dataset.vi;renderExplorer();showView("answers");});
  renderExplorer();
  if (run.error) notify(run.error);
}
function renderExplorer() {
  const r=state.run;if(!r)return;
  state.selectedQ=Math.min(state.selectedQ,r.config.problems.length-1);state.selectedV=Math.min(state.selectedV,r.config.variants.length-1);
  $("question-select").innerHTML=r.config.problems.map((p,i)=>`<option value="${i}">${i+1}. ${esc(p.title)}</option>`).join("");
  $("variant-select").innerHTML=r.config.variants.map((v,i)=>`<option value="${i}">${esc(label(v))}</option>`).join("");
  $("question-select").value=state.selectedQ;$("variant-select").value=state.selectedV;
  const b=r.baselines[state.selectedQ],c=r.comparisons[`${state.selectedV}:${state.selectedQ}`];
  const problem=r.config.problems[state.selectedQ];
  $("active-question").textContent=problem.title;
  $("problem-statement").textContent=problem.description;
  $("baseline-answer").textContent=b?.execution.code || b?.text || "Waiting for the original Python solution…";
  $("pruned-answer").textContent=c?.answer.execution.code || c?.answer.text || "Waiting for the pruned Python solution…";
  $("baseline-answer").classList.toggle("waiting",!b);$("pruned-answer").classList.toggle("waiting",!c);
  const timing=answer=>answer?`${answer.seconds.toFixed(1)} s · ${answer.tokens||"?"} tokens${answer.truncated?" · context limit reached":""}`:"";
  $("baseline-time").textContent=timing(b);$("pruned-time").textContent=timing(c?.answer);
  $("pruned-label").textContent=label(r.config.variants[state.selectedV]).toUpperCase();
  $("outcome-score").textContent=formatted(c?.score);
  $("outcome-verdict").textContent=c?(c.score===1?"Same tested outcomes":"Different outcomes / execution failure"):"Waiting for execution";
  $("outcome-reason").textContent=c?.reason || "Both programs will run on identical inputs. No LLM judge is used.";
  const be=b?.execution,pe=c?.answer.execution;
  const errors=be?.error&&be.error===pe?.error?`Original and pruned: ${be.error}`:[be?.error?`Original: ${be.error}`:"",pe?.error?`Pruned: ${pe.error}`:""].filter(Boolean).join(" ");
  const guidance=r.model==="gpt2"&&(be?.status==="invalid"||pe?.status==="invalid")&&!errors.includes("Qwen2.5-Coder")?" GPT-2 is not coding/instruction tuned. Choose Qwen2.5-Coder for more useful solutions to the same tested questions.":"";
  $("correctness").textContent=`Expected-test correctness: original ${be?`${be.passed}/${be.total}`:"pending"} · pruned ${pe?`${pe.passed}/${pe.total}`:"pending"}. ${errors}${guidance}`;
  const output=(execution,i)=>{
    if(execution?.cases[i])return JSON.stringify(execution.cases[i].output);
    if(!execution)return "Pending";
    return ({invalid:"Not run — invalid or incomplete solution",timeout:"No result — execution timed out",cancelled:"No result — execution cancelled",error:"No result — execution failed"})[execution.status]||"No result";
  };
  $("case-results").innerHTML=`<thead><tr><th>INPUT ARGS</th><th>EXPECTED</th><th>ORIGINAL OUTPUT</th><th>PRUNED OUTPUT</th></tr></thead><tbody>${problem.tests.map((test,i)=>`<tr><td><code>${esc(JSON.stringify(test.args))}</code></td><td><code>${esc(JSON.stringify(test.expected))}</code></td><td><code>${esc(output(be,i))}</code></td><td><code>${esc(output(pe,i))}</code></td></tr>`).join("")}</tbody>`;
  renderSelectedPipeline();
}

function selectPipelinePair(question, variant) {
  state.selectedQ=question;state.selectedV=variant;
  if(state.run?.config.problems)renderExplorer();else renderSelectedPipeline();
}

async function renderSelectedPipeline() {
  const run=state.run;if(!run)return;
  const qi=run.config.problems?state.selectedQ:0,vi=state.selectedV;
  const baseline=run.baselines[qi],pruned=run.comparisons[`${vi}:${qi}`]?.answer;
  const draw=()=>window.gpt2View.render(run,true,qi,vi,selectPipelinePair);
  draw();
  if(["running","queued","cancelling"].includes(run.status)||!baseline?.pipeline_available||!pruned?.pipeline_available)return;
  if(baseline.pipeline&&pruned.pipeline)return;
  const key=`${run.id}:${run.revision}:${qi}:${vi}`;
  if(!pipelineCache.has(key)){
    const pending=api(`/api/runs/${run.id}/pipeline?question=${qi}&variant=${vi}`);
    pipelineCache.set(key,pending);
    if(pipelineCache.size>4)pipelineCache.delete(pipelineCache.keys().next().value);
  }
  try{
    const pair=await pipelineCache.get(key);
    if(state.run!==run||state.selectedQ!==qi||state.selectedV!==vi)return;
    // Keep only the selected pair on the live result record. The small cache
    // enables back-and-forth inspection without retaining every tensor sample.
    Object.values(run.baselines).forEach(answer=>delete answer.pipeline);
    Object.values(run.comparisons).forEach(cell=>delete cell.answer.pipeline);
    baseline.pipeline=pair.baseline;pruned.pipeline=pair.pruned;
    draw();
  }catch(error){
    pipelineCache.delete(key);
    baseline.pipeline_error=`Could not load this question's pipeline: ${error.message}`;
    if(state.run?.id===run.id&&state.selectedQ===qi&&state.selectedV===vi)draw();
  }
}
async function history() {
  const runs=(await api("/api/runs")).filter(run=>run.kind==="experiment"&&run.model==="Qwen/Qwen2.5-Coder-1.5B-Instruct"); const current=state.run?.id;
  $("history").innerHTML='<option value="">Previous experiments</option>'+runs.map(r=>`<option value="${esc(r.id)}">${esc(r.topic)} · ${esc(r.status)} · ${new Date(r.created_at*1000).toLocaleTimeString()}</option>`).join("");
  if(current)$("history").value=current;
}
async function poll(id) {
  try {
    const r=await api(`/api/runs/${id}`);
    if(["experiment","text-generation"].includes(r.kind) && r.revision!==state.revision){state.revision=r.revision;renderResults(r);}
    if(["queued","running","cancelling"].includes(r.status)){state.pollTimer=setTimeout(()=>poll(id),1000);return;}
    state.active=null;setBusy(false);
    if(r.error)notify(r.error);
    else if(r.status==="cancelled")notify("Task stopped. Finished answers are retained; remaining cells are incomplete.");
    await history();
  } catch(error) {
    notify(`Connection interrupted: ${error.message}. Reconnecting in 3 seconds; the server task may still be running.`);
    state.pollTimer=setTimeout(()=>poll(id),3000);
  }
}
$("add-variant").onclick=()=>addVariant();
$("model").onchange=()=>modelNote();
$("templates").onclick=async()=>{try{notify("");const data=await api(`/api/problems?category=${encodeURIComponent($("topic").value)}`);state.bank=data.problems.map(problem=>({problem,selected:true}));renderBank();}catch(e){notify(e.message);}};
$("custom-form").onsubmit=event=>{event.preventDefault();if(state.bank.length>=20)return notify("Remove a problem first; at most 20 per run.");try{const problem=parseProblem($("custom-question").value);if(state.bank.some(q=>q.problem.id===problem.id))throw new Error("Problem ID is already present.");state.bank.push({problem,selected:true});$("custom-question").value="";renderBank();$("bank-details").open=true;}catch(e){notify(e.message);}};
$("generate-question").onclick=async()=>{if(state.bank.length>=20)return notify("Remove a problem first; at most 20 per run.");try{notify("");const problem=await api("/api/problems/new",{category:$("topic").value,existing:state.bank.map(q=>q.problem.id)});state.bank.push({problem,selected:true});renderBank();$("bank-details").open=true;notify("New Python problem added from the validated local catalog. Review its tests before running.");}catch(e){notify(e.message);}};
$("run-form").onsubmit=async event=>{
  event.preventDefault();notify("");
  const variants=getVariants(),selected=state.bank.filter(q=>q.selected),problems=selected.map(q=>q.problem);
  if(!variants.length)return notify("Select at least one pruning variant.");
  if(selected.some(q=>q.error))return notify("Correct invalid JSON in the selected problem editor first.");
  if(!problems.length)return notify("Select at least one Python problem.");
  if(!$("execution-consent").checked)return notify("Confirm restricted local Python execution before running.");
  try{
    const payload={model_name:$("model").value,topic:$("topic").value,problems,variants,execution_consent:true,seed:+$("seed").value};
    const response=await api("/api/runs",payload);
    state.active=response.id;state.revision=-1;state.selectedQ=state.selectedV=0;setBusy(true);showView("overview");poll(response.id);
  }catch(e){notify(e.message);}
};
$("cancel").onclick=async()=>{if(state.active)try{await api(`/api/runs/${state.active}/cancel`,{});}catch(e){notify(e.message);}};
$("question-select").onchange=()=>{state.selectedQ=+$("question-select").value;renderExplorer();};
$("variant-select").onchange=()=>{state.selectedV=+$("variant-select").value;renderExplorer();};
$("tab-overview").onclick=()=>showView("overview");$("tab-answers").onclick=()=>showView("answers");
$("history").onchange=async()=>{if(!$("history").value)return;try{notify("");state.selectedQ=state.selectedV=0;const run=await api(`/api/runs/${$("history").value}`);loadDraft(run);renderResults(run);}catch(e){notify(e.message);}};
$("export").onclick=async()=>{if(!state.run)return;try{const run=await api(`/api/runs/${state.run.id}?include_pipelines=true`);const url=URL.createObjectURL(new Blob([JSON.stringify(run,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=`pruning-lab-${run.id}.json`;a.click();URL.revokeObjectURL(url);}catch(error){notify(`Could not export results: ${error.message}`);}};
async function init(){
  try{
    const config=await api("/api/config");state.token=config.token;$("hardware").textContent=`${config.device} · FP16 on CUDA / FP32 on CPU`;
    state.bank=config.problems.map(problem=>({problem,selected:true}));renderBank();
    addVariant("layers",1);addVariant("heads",3);addVariant("magnitude",20);modelNote();
    await history();
    if(config.active_run){state.active=config.active_run;setBusy(true);poll(config.active_run);}
    if(config.busy_elsewhere)notify("The GPT-2 example is preparing automatically. Wait for it to finish before starting a Qwen experiment.");
  }catch(e){notify(`Could not initialize the local server: ${e.message}. Refresh after starting run_web.py.`);$("run").disabled=true;}
}
init();
