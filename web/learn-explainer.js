/* Local pruning controls and measured inspector for the actual upstream
 * Transformer Explainer Svelte/D3 visualization in /static/transformer/. */
"use strict";
window.gpt2View=(()=>{
  const root=()=>document.getElementById("gpt2-view");
  const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const token=value=>String(value).replace(/ /g,"␠").replace(/\n/g,"↵").replace(/\t/g,"⇥");
  const pct=value=>`${(100*value).toFixed(2)}%`;
  const num=value=>Number(value).toFixed(4);
  const absMax=values=>Math.max(.000001,...values.flat(Infinity).map(Math.abs));
  const state={run:null,id:null,variant:0,view:"original",layer:0,head:0,query:-1,key:0,drawer:null,matrix:"attention",frame:""};
  const tones={embedding:[170,182,198],q:[126,171,227],k:[225,153,154],v:[125,190,164],out:[169,148,215],mlp:[167,161,213]};
  function color(value,scale,tone=tones.out){const t=Math.min(1,Math.abs(value)/(scale||1));return `rgb(${[250,251,253].map((v,i)=>Math.round(v+(tone[i]-v)*t)).join(",")})`;}
  function strip(values,scale,tone=tones.out){
    return `<svg class="learn-vector-strip" viewBox="0 0 ${values.length*54} 28" role="img" aria-label="Measured vector sample">${values.map((v,i)=>`<g><rect x="${i*54}" width="52" height="27" rx="2" fill="${color(v,scale,tone)}"/><text x="${i*54+26}" y="17" text-anchor="middle">${num(v)}</text><title>Dimension ${i}: ${num(v)}</title></g>`).join("")}</svg>`;
  }
  function pair(){
    const run=state.run,a=run?.baselines?.[0]?.pipeline,b=run?.comparisons?.[`${state.variant}:0`]?.answer?.pipeline;
    if(!a||!b||a.prompt!==run.config.prompt||b.prompt!==run.config.prompt||a.tokens.map(t=>t.id).join()!==b.tokens.map(t=>t.id).join())return null;
    return {a,b,p:state.view==="original"?a:b};
  }
  function indexControl(axis,label,count){
    const value=state[axis];
    return `<div class="learn-index-control" data-learn-scroll="${axis}" title="Scroll up or down to change ${axis}"><label for="learn-${axis}">${label}</label><div class="learn-index-picker"><select id="learn-${axis}">${Array.from({length:count},(_,i)=>`<option value="${i}"${value===i?" selected":""}>${i+1} / ${count}</option>`).join("")}</select><span class="learn-index-arrows">${[-1,1].map(delta=>`<button type="button" data-learn-axis="${axis}" data-learn-step="${delta}" data-learn-nav="toolbar-${axis}-${delta}" aria-label="${delta<0?"Previous":"Next"} ${axis}"${value+delta<0||value+delta>=count?" disabled":""}>${delta<0?"▴":"▾"}</button>`).join("")}</span></div></div>`;
  }
  function matrixCells(p,data,mode){
    const values=data[mode==="attention"?"attention":"scores"][state.head],scale=absMax(data.scores);
    return values.flatMap((row,i)=>row.map((v,j)=>{
      const blocked=j>i&&mode!=="scores",selected=i===state.query&&j===state.key;
      const fill=blocked?"#edf0f3":mode==="attention"?color(v,1,[212,139,112]):color(v,scale,v<0?[129,165,208]:[221,160,135]);
      return `<g role="button" tabindex="0" data-learn-cell="${i},${j}" aria-label="Query ${i}, key ${j}: ${blocked?"future token blocked":mode==="attention"?pct(v):num(v)}"><rect x="${26+j*20}" y="${25+i*20}" width="19" height="19" rx="1" fill="${fill}" class="${selected?"matrix-focus":""}"/><title>${esc(token(p.tokens[i].text))} → ${esc(token(p.tokens[j].text))}: ${blocked?"causally blocked":mode==="attention"?pct(v):num(v)}</title></g>`;
    })).join("");
  }
  function inspector(a,b,p){
    if(!state.drawer)return "";
    const data=p.layers[state.layer],la=a.layers[state.layer],lb=b.layers[state.layer];
    const row=(label,values,field,tone)=>`<div class="learn-tensor-row"><span>${label}</span>${strip(values,Math.max(absMax(la[field]),absMax(lb[field])),tone)}</div>`;
    let heading="",body="";
    if(state.drawer==="embeddings"){
      heading="Embedding";body='<p>Token embedding + position embedding = model input. Pruning presets leave these vectors unchanged. These are eight sampled dimensions.</p>';
      for(const [field,label] of [["token","Token embedding"],["position","Position embedding"],["combined","Combined input"]])body+=`<div class="learn-tensor-row"><span>${label}</span>${strip(p.embeddings[field][state.query],Math.max(absMax(a.embeddings[field]),absMax(b.embeddings[field])),tones.embedding)}</div>`;
    }else if(state.drawer==="qkv"){
      heading="Query, Key and Value";body='<p>The normalized representation is projected into Q, K and V. Q and K determine attention scores; V supplies the content to combine.</p>';
      for(const field of ["q","k","v"])body+=row(`${field.toUpperCase()} · token ${state.query}`,data[field][state.head][state.query],field,tones[field]);
    }else if(state.drawer==="attention"){
      heading="Masked self-attention";
      const blocked=state.key>state.query;
      body=`<p>Q·K / √${p.head_dim} → causal mask → softmax. Query ${state.query} (${esc(token(p.tokens[state.query].text))}) → key ${state.key} (${esc(token(p.tokens[state.key].text))}): score <b>${num(data.scores[state.head][state.query][state.key])}</b>; attention <b>${blocked?"0% · future token blocked":pct(data.attention[state.head][state.query][state.key])}</b>. ${data.masked_heads.includes(state.head)?"This head's attention is computed, but its output contribution is masked to zero.":""}</p>`;
      body+=`<div class="learn-matrix-modes">${[["scores","QKᵀ / √d"],["masked","Causal mask"],["attention","Softmax"]].map(([id,label])=>`<button type="button" data-learn-matrix="${id}" aria-pressed="${state.matrix===id}">${label}</button>`).join("")}</div>`;
      const size=p.tokens.length*20+28;
      body+=`<svg class="learn-attention-matrix" viewBox="0 0 ${size} ${size}" aria-label="Measured attention matrix">${p.tokens.map((_,i)=>`<text x="${32+i*20}" y="14">${i}</text><text x="8" y="${36+i*20}">${i}</text>`).join("")}${matrixCells(p,data,state.matrix)}</svg>`;
      body+=row("Actual head contribution",data.head_output[state.head][state.query],"head_output",tones.out);
    }else if(state.drawer==="mlp"){
      heading="MLP and residual stream";body=`<p>Linear expansion → GELU → linear projection. The MLP output is added to the residual stream. ${data.masked_weights?`${data.masked_weights.toLocaleString()} of ${data.total_mlp_weights.toLocaleString()} MLP parameters are masked in this layer.`:"No MLP weight mask is applied in this view."} ${data.bypassed?"This block is bypassed: its internally computed values are discarded, and its effective output equals its input.":""}</p>`;
      body+=row("MLP output",data.mlp_output[state.query],"mlp_output",tones.mlp)+row("Block input",data.input[state.query],"input",tones.embedding)+row("Effective block output",data.output[state.query],"output",tones.out);
    }else if(state.drawer==="blocks"){
      heading="Transformer blocks";body='<p>GPT-2 repeats attention and MLP computations across twelve layers. Input embeddings are shared; inspecting a later layer does not skip earlier layers in the actual model.</p>';
      body+=`<div class="learn-matrix-modes">${p.layers.map((l,i)=>`<button type="button" data-learn-layer="${i}" aria-pressed="${state.layer===i}">L${i+1}${l.bypassed?" · bypassed":l.masked_heads.length?" · heads masked":l.masked_weights?" · weights masked":""}</button>`).join("")}</div>`;
    }else{
      heading="Next-token probabilities";body=`<p>Final layer norm → vocabulary projection → softmax. Predictions after token ${state.query}: ${esc(token(p.tokens[state.query].text))}. The last input token predicts the start of the continuation. Earlier selections inspect earlier prefixes. Top-eight probabilities are not renormalized; other vocabulary tokens retain ${pct(p.predictions[state.query].other_probability)} of the probability mass. The expanded diagram labels log probabilities as log p; they are not raw model logits.</p>`;
    }
    return `<section class="learn-inspector" aria-label="Expanded ${esc(heading)} details"><header><h3>${heading} · ${state.view} · layer ${state.layer+1}, head ${state.head+1}</h3><button type="button" data-learn-close>Close ×</button></header>${body}</section>`;
  }
  function completion(answer,label){
    const text=answer.text||"",snippet=text.slice(0,160),stop=answer.truncated?"context exhausted":answer.stop_reason==="eos"?"end of text":answer.stop_reason||"finished";
    return `<article><h3>${label}</h3><p class="learn-output-preview">${esc(snippet)}${text.length>160?" …":""}</p><details><summary>Full continuation · ${answer.tokens} tokens · ${esc(stop)}</summary><div>${esc(text)}</div></details></article>`;
  }
  function updateDiagram(){
    const current=pair(),frame=document.getElementById("learn-network-frame");
    if(!current||!frame)return;
    const width=Math.max(1600,root().querySelector(".learn-canvas-scroll").clientWidth);
    frame.contentWindow.postMessage({type:"llm-visualizer:update",payload:{a:current.a,b:current.b,view:state.view,layer:state.layer,head:state.head,query:state.query,width,topology:state.view==="original"?state.run.baseline_view:state.run.pruning_views?.[state.variant]}},location.origin);
  }
  function paint(){
    const run=state.run,current=pair();
    if(!run||["running","queued","cancelling"].includes(run.status)){root().innerHTML='<div class="learn-loading">Loading the prepared transformer…</div>';return;}
    if(!current){root().innerHTML=`<div class="learn-loading">${esc(run.baselines?.[0]?.pipeline_error||"Loading the measured original and pruned states…")}</div>`;return;}
    const {a,b,p}=current,preset=run.walkthrough_presets[state.variant];
    state.layer=Math.max(0,Math.min(state.layer,p.layers.length-1));state.head=Math.max(0,Math.min(state.head,p.head_count-1));
    if(state.query<0||state.query>=p.tokens.length)state.query=p.tokens.length-1;
    state.key=Math.min(state.key,p.tokens.length-1);
    if(!document.getElementById("learn-network-frame")){
      root().innerHTML='<div class="learn-toolbar-slot"></div><div class="learn-canvas-scroll" tabindex="0" aria-label="Transformer network; scroll horizontally on small screens"><iframe id="learn-network-frame" class="learn-network-frame" src="/static/transformer/index.html" title="Transformer Explainer visualization with measured pruning" height="800"></iframe></div><div class="learn-diagram-caption"><span>Original Transformer Explainer components · scroll Layer / Head or use the arrows above each block.</span><span>Saved forward pass, not every generated token. Vector textures are measured samples; unrecorded MLP intermediates are architectural.</span></div><div class="learn-query-picker"></div><div class="learn-inspector-slot"></div><section class="learn-output-comparison" aria-label="Original and pruned text output"></section>';
      const frame=document.getElementById("learn-network-frame");
      frame.width=String(Math.max(1600,root().querySelector(".learn-canvas-scroll").clientWidth));
      frame.addEventListener("load",updateDiagram);
    }
    const short={layers:"The last block's contribution is bypassed.",heads:"Three head outputs per layer are zeroed.",magnitude:"20% of the smallest MLP parameters are zeroed.",random:"About 20% of MLP parameters are zeroed at seeded random positions."}[preset.technique];
    root().querySelector(".learn-toolbar-slot").innerHTML=`<div class="learn-toolbar"><div class="learn-state-switch" role="group" aria-label="Model state"><button type="button" data-model-state="original" aria-pressed="${state.view==="original"}">Original</button><button type="button" data-model-state="pruned" aria-pressed="${state.view==="pruned"}">Pruned</button></div>${indexControl("layer","Layer",p.layers.length)}${indexControl("head","Head",p.head_count)}<span class="learn-scroll-hint">Scroll Layer / Head to explore</span><span class="learn-preset-note"><strong>${state.view==="original"?"Original model shown. Pruned preset:":"Pruned model shown."}</strong> ${esc(short)}</span></div>`;
    root().querySelector(".learn-query-picker").innerHTML=`<span>Inspect token</span>${p.tokens.map((t,i)=>`<button type="button" data-learn-token="${i}" aria-pressed="${state.query===i}" title="Position ${i} · token ID ${t.id}">${esc(token(t.text))}</button>`).join("")}`;
    root().querySelector(".learn-inspector-slot").innerHTML=inspector(a,b,p);
    root().querySelector(".learn-output-comparison").innerHTML=completion(run.baselines[0],"Original completion · preview")+completion(run.comparisons[`${state.variant}:0`].answer,`Pruned completion · ${esc(preset.title)}`);
    updateDiagram();
  }
  function move(axis,delta){
    const current=pair();if(!current||!["layer","head"].includes(axis))return false;
    const count=axis==="layer"?current.p.layers.length:current.p.head_count;
    const value=Math.max(0,Math.min(count-1,state[axis]+delta));
    if(value===state[axis])return false;
    state[axis]=value;return true;
  }
  function restoreFocus(selector){
    if(!selector)return;
    const item=root().querySelector(selector);if(!item)return;
    if(item.disabled)document.getElementById(`learn-${item.dataset.learnAxis}`)?.focus({preventScroll:true});
    else item.focus({preventScroll:true});
  }
  function render(run,visible,question=0,variant=0,onSelect=null){
    root().hidden=!visible;if(!visible)return;
    if((run?.id??null)!==state.id){const preset=run?.walkthrough_presets?.[variant];state.id=run?.id??null;state.view="original";state.layer=preset?.inspection_layer??0;state.head=preset?.inspection_head??0;state.query=-1;state.key=0;state.drawer=null;state.frame="";}
    if(run&&variant!==state.variant){const preset=run.walkthrough_presets[variant];state.view="pruned";state.layer=preset.inspection_layer;state.head=preset.inspection_head;}
    state.run=run;state.variant=variant;
    const frame=`${run?.id}:${run?.revision}:${variant}:${!!run?.baselines?.[0]?.pipeline}:${!!run?.comparisons?.[`${variant}:0`]?.answer?.pipeline}:${run?.baselines?.[0]?.pipeline_error||""}`;
    if(frame===state.frame)return;state.frame=frame;paint();
  }
  document.addEventListener("DOMContentLoaded",()=>{
    root().addEventListener("click",activate);root().addEventListener("keydown",activate);
    function activate(event){
      const item=event.target.closest("[data-model-state],[data-learn-token],[data-learn-cell],[data-learn-close],[data-learn-layer],[data-learn-matrix],[data-learn-step]");if(!item)return;
      if(event.type==="keydown"&&event.key!=="Enter"&&event.key!==" ")return;
      if(item.disabled)return;
      event.preventDefault();let focus=null;
      if(item.dataset.learnStep!==undefined){if(!move(item.dataset.learnAxis,+item.dataset.learnStep))return;focus=`[data-learn-nav="${item.dataset.learnNav}"]`;}
      else if(item.dataset.modelState){state.view=item.dataset.modelState;focus=`[data-model-state="${state.view}"]`;}
      else if(item.hasAttribute("data-learn-close")){state.drawer=null;focus=`[data-model-state="${state.view}"]`;}
      else if(item.dataset.learnToken!==undefined){state.query=+item.dataset.learnToken;state.key=Math.min(state.key,state.query);focus=`[data-learn-token="${state.query}"]`;}
      else if(item.dataset.learnCell){[state.query,state.key]=item.dataset.learnCell.split(",").map(Number);state.drawer="attention";}
      else if(item.dataset.learnLayer!==undefined){state.layer=+item.dataset.learnLayer;focus=`[data-learn-layer="${state.layer}"]`;}
      else if(item.dataset.learnMatrix){state.matrix=item.dataset.learnMatrix;focus=`[data-learn-matrix="${state.matrix}"]`;}
      paint();restoreFocus(focus);
    }
    root().addEventListener("change",event=>{const id=event.target.id;if(id==="learn-layer")state.layer=+event.target.value;else if(id==="learn-head")state.head=+event.target.value;else return;paint();document.getElementById(id)?.focus({preventScroll:true});});
    let wheelAxis=null,wheelDelta=0,wheelTime=0,lastStep=-Infinity;
    root().addEventListener("wheel",event=>{
      const control=event.target.closest("[data-learn-scroll]");
      if(!control||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey||Math.abs(event.deltaX)>Math.abs(event.deltaY)||!event.deltaY)return;
      event.preventDefault();
      const axis=control.dataset.learnScroll,now=performance.now();
      if(axis!==wheelAxis||now-wheelTime>250||Math.sign(wheelDelta)!==Math.sign(event.deltaY))wheelDelta=0;
      wheelAxis=axis;wheelTime=now;wheelDelta+=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?window.innerHeight:1);
      if(Math.abs(wheelDelta)<24||now-lastStep<120)return;
      const delta=Math.sign(wheelDelta);wheelDelta=0;if(!move(axis,delta))return;lastStep=now;
      const active=document.activeElement,focus=active?.id?`#${active.id}`:active?.dataset.learnNav?`[data-learn-nav="${active.dataset.learnNav}"]`:null;
      paint();restoreFocus(focus);
    },{passive:false});
    window.addEventListener("message",event=>{
      const frame=document.getElementById("learn-network-frame");
      if(!frame||event.origin!==location.origin||event.source!==frame.contentWindow)return;
      const message=event.data;if(!message||typeof message!=="object")return;
      if(message.type==="llm-visualizer:ready")updateDiagram();
      else if(message.type==="llm-visualizer:size"&&Number.isFinite(message.height)&&Number.isFinite(message.width)){
        frame.height=String(Math.max(500,Math.min(2200,Math.ceil(message.height))));
        frame.width=String(Math.max(1600,Math.min(6000,Math.ceil(message.width))));
      }
      else if(message.type==="llm-visualizer:navigate"&&[-1,1].includes(message.delta)){if(move(message.axis,message.delta))paint();}
      else if(message.type==="llm-visualizer:inspect"&&["embeddings","qkv","attention","mlp","blocks","probabilities"].includes(message.drawer)){state.drawer=message.drawer;paint();}
    });
    let lastWidth=0;
    const resize=new ResizeObserver(()=>{const width=root().clientWidth;if(width!==lastWidth){lastWidth=width;updateDiagram();}});
    resize.observe(root());
  });
  return {render};
})();
