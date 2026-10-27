/*
 * Learn-only adaptation of Transformer Explainer's continuous-column layout:
 * Embedding → QKV → multi-head attention → MLP → remaining blocks → softmax.
 * Upstream: https://github.com/poloclub/transformer-explainer (MIT; bundled notice).
 * Rendering uses our cached PyTorch measurements, not the upstream ONNX model.
 */
"use strict";
window.gpt2View=(()=>{
  const root=()=>document.getElementById("gpt2-view");
  const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const token=value=>String(value).replace(/ /g,"␠").replace(/\n/g,"↵").replace(/\t/g,"⇥");
  const pct=value=>`${(100*value).toFixed(2)}%`;
  const num=value=>Number(value).toFixed(4);
  const absMax=values=>Math.max(.000001,...values.flat(Infinity).map(Math.abs));
  const state={run:null,id:null,variant:0,view:"original",layer:0,head:0,query:-1,key:0,drawer:null,matrix:"attention",frame:"",select:null};
  const tones={embedding:[170,182,198],q:[126,171,227],k:[225,153,154],v:[125,190,164],out:[169,148,215],mlp:[167,161,213]};
  function color(value,scale,tone=tones.out){const t=Math.min(1,Math.abs(value)/(scale||1));return `rgb(${[250,251,253].map((v,i)=>Math.round(v+(tone[i]-v)*t)).join(",")})`;}
  function vector(values,x,y,width=12,height=22,scale=1,tone=tones.out){
    const h=height/values.length;
    return `<g>${values.map((value,i)=>`<rect x="${x}" y="${y+i*h}" width="${width}" height="${Math.max(.5,h-.25)}" fill="${color(value,scale,tone)}"><title>Sample dimension ${i}: ${num(value)}</title></rect>`).join("")}<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="none" stroke="#d7dce4" stroke-width=".5"/></g>`;
  }
  function strip(values,scale,tone=tones.out){
    return `<svg class="learn-vector-strip" viewBox="0 0 ${values.length*54} 28" role="img" aria-label="Measured vector sample">${values.map((v,i)=>`<g><rect x="${i*54}" width="52" height="27" rx="2" fill="${color(v,scale,tone)}"/><text x="${i*54+26}" y="17" text-anchor="middle">${num(v)}</text><title>Dimension ${i}: ${num(v)}</title></g>`).join("")}</svg>`;
  }
  function pair(){
    const run=state.run,a=run?.baselines?.[0]?.pipeline,b=run?.comparisons?.[`${state.variant}:0`]?.answer?.pipeline;
    if(!a||!b)return null;
    if(a.prompt!==run.config.prompt||b.prompt!==run.config.prompt||a.tokens.map(t=>t.id).join()!==b.tokens.map(t=>t.id).join())return null;
    return {a,b,p:state.view==="original"?a:b};
  }
  function route(x1,y1,x2,y2,kind="",selected=false){return `<path d="M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}" class="flow-route ${kind}${selected?" selected-route":""}"/>`;}
  function indexControl(axis,label,count){
    const value=state[axis];
    return `<div class="learn-index-control" data-learn-scroll="${axis}" title="Scroll up or down to change ${axis}"><label for="learn-${axis}">${label}</label><div class="learn-index-picker"><select id="learn-${axis}">${Array.from({length:count},(_,i)=>`<option value="${i}"${value===i?" selected":""}>${i+1} / ${count}</option>`).join("")}</select><span class="learn-index-arrows">${[-1,1].map(delta=>`<button type="button" data-learn-axis="${axis}" data-learn-step="${delta}" data-learn-nav="toolbar-${axis}-${delta}" aria-label="${delta<0?"Previous":"Next"} ${axis}"${value+delta<0||value+delta>=count?" disabled":""}>${delta<0?"▴":"▾"}</button>`).join("")}</span></div></div>`;
  }
  function layerNavigation(x,drawer,label,count){
    return `<g class="learn-layer-navigation" transform="translate(${x},0)" data-learn-scroll="layer" role="group" aria-label="Layer navigation for ${esc(label)}"><title>Scroll up or down to change layer</title>${[-1,1].map(delta=>{
      const disabled=state.layer+delta<0||state.layer+delta>=count;
      return `<g class="learn-layer-arrow" transform="translate(${delta<0?0:130},1)" role="button" tabindex="${disabled?-1:0}" aria-disabled="${disabled}" aria-label="${delta<0?"Previous":"Next"} layer in ${esc(label)}" data-learn-axis="layer" data-learn-step="${delta}" data-learn-nav="${drawer}-${delta}"><title>${delta<0?"Previous":"Next"} layer</title><rect width="26" height="22" rx="4"/><path d="${delta<0?"M15 7 l-4 4 4 4":"M11 7 l4 4 -4 4"}"/></g>`;
    }).join("")}<text x="78" y="16" text-anchor="middle" aria-live="polite">Layer ${state.layer+1} / ${count}</text></g>`;
  }
  function title(label,x,drawer,subtitle="",layerCount=0){
    return `${layerCount?layerNavigation(x,drawer,label,layerCount):""}<g role="button" tabindex="0" data-learn-open="${drawer}" aria-label="Inspect ${esc(label)}"><rect x="${x-7}" y="24" width="175" height="51" fill="transparent"/><text class="column-heading" x="${x}" y="45">${esc(label)} ⊕</text>${subtitle?`<text class="column-subtitle" x="${x}" y="65">${esc(subtitle)}</text>`:""}</g>`;
  }
  function matrixCells(p,data,x,y,cell,mode="attention"){
    const values=data[mode==="attention"?"attention":"scores"][state.head],scale=absMax(data.scores);
    return values.flatMap((row,i)=>row.map((v,j)=>{
      const blocked=j>i&&mode!=="scores",selected=i===state.query&&j===state.key;
      const fill=blocked?"#edf0f3":mode==="attention"?color(v,1,[212,139,112]):color(v,scale,v<0?[129,165,208]:[221,160,135]);
      return `<g role="button" tabindex="0" data-learn-cell="${i},${j}" aria-label="Query ${i}, key ${j}: ${blocked?"future token blocked":mode==="attention"?pct(v):num(v)}"><rect x="${x+j*cell}" y="${y+i*cell}" width="${cell-1}" height="${cell-1}" rx="1" fill="${fill}" class="${selected?"matrix-focus":""}"/><title>${esc(token(p.tokens[i].text))} → ${esc(token(p.tokens[j].text))}: ${blocked?"causally blocked":mode==="attention"?pct(v):num(v)}</title></g>`;
    })).join("");
  }
  function graph(a,b,p){
    const run=state.run,data=p.layers[state.layer],la=a.layers[state.layer],lb=b.layers[state.layer];
    const n=p.tokens.length,y0=112,gap=29,row=i=>y0+i*gap,height=Math.max(480,row(n-1)+112),mid=(row(0)+row(n-1))/2;
    const scales={};for(const field of ["q","k","v","head_output","mlp_output","output"])scales[field]=Math.max(absMax(la[field]),absMax(lb[field]));
    const embeddingScale=Math.max(absMax(a.embeddings.combined),absMax(b.embeddings.combined));
    const bypassed=data.bypassed,masked=data.masked_heads.includes(state.head);
    const cell=Math.min(18,200/n),matrixX=481,matrixY=mid-n*cell/2;
    const tokenRows=p.tokens.map((t,i)=>`<g data-learn-token="${i}" role="button" tabindex="0" aria-label="Inspect token ${i}: ${esc(token(t.text))}"><rect class="${i===state.query?"token-selected":"token-hit"}" x="10" y="${row(i)-15}" width="149" height="26" rx="4"/><text class="token-name" x="133" y="${row(i)+3}" text-anchor="end">${esc(token(t.text))}</text>${vector(p.embeddings.combined[i],146,row(i)-12,11,22,embeddingScale,tones.embedding)}</g>`).join("");
    const embeddedPaths=p.tokens.map((_,i)=>[0,1,2].map((j)=>route(158,row(i)-1,285+j*34,row(i)-1,["query-route","key-route","value-route"][j],i===state.query)).join("")).join("");
    const qkv=p.tokens.map((_,i)=>["q","k","v"].map((kind,j)=>vector(data[kind][state.head][i],285+j*34,row(i)-12,11,22,scales[kind],tones[kind])).join("")).join("");
    const attentionPaths=p.tokens.map((_,i)=>route(365,row(i)-1,matrixX-9,matrixY+i*cell+cell/2,"",i===state.query)).join("");
    const headOutputs=p.tokens.map((_,i)=>`${masked?`<path class="pruned-route" d="M${matrixX+n*cell+12},${mid} C725,${mid} 730,${row(i)} 766,${row(i)}"/>`:route(matrixX+n*cell+12,mid,766,row(i),"",i===state.query)}${vector(data.head_output[state.head][i],766,row(i)-12,11,22,scales.head_output,tones.out)}`).join("");
    const mlp=p.tokens.map((_,i)=>`${route(778,row(i),847,row(i),"",i===state.query)}<circle class="operation-node" cx="817" cy="${row(i)}" r="6"/><text x="813" y="${row(i)+4}">+</text><path class="architecture-shape" d="M848,${row(i)-10} L912,${row(i)-15} L934,${row(i)-15} L997,${row(i)-10} V${row(i)+10} L934,${row(i)+15} L912,${row(i)+15} L848,${row(i)+10} Z"/>${vector(data.mlp_output[i],998,row(i)-12,11,22,scales.mlp_output,tones.mlp)}<path class="residual-route" d="M832,${row(i)} V${row(i)-20} H1034 V${row(i)}"/><circle class="operation-node" cx="1034" cy="${row(i)}" r="6"/><text x="1030" y="${row(i)+4}">+</text>`).join("");
    const output=p.tokens.map((_,i)=>`${route(1040,row(i),1059,row(i),"",i===state.query)}${vector(data.output[i],1059,row(i)-12,11,22,scales.output,tones.out)}`).join("");
    const left=p.layers.length-state.layer-1,following=p.layers.slice(state.layer+1).filter(l=>l.bypassed).length;
    const rest=left?`<g data-learn-open="blocks" role="button" tabindex="0" aria-label="Inspect transformer layers"><rect class="other-block" x="1138" y="${mid-46}" width="55" height="96" rx="5"/><rect class="other-block" x="1130" y="${mid-39}" width="55" height="96" rx="5"/><rect class="other-block" x="1122" y="${mid-32}" width="55" height="96" rx="5"/><text x="1148" y="${mid+4}" text-anchor="middle">+${left}</text><text class="column-subtitle" x="1148" y="${mid+21}" text-anchor="middle">blocks</text>${following?`<text class="bypass-label" x="1148" y="${mid+82}" text-anchor="middle">${following} bypassed</text>`:""}</g>`:`<rect class="other-block" x="1122" y="${mid-18}" width="69" height="37" rx="5"/><text x="1156" y="${mid+4}" text-anchor="middle">Final norm</text>`;
    const prediction=p.predictions[state.query],candidates=prediction.candidates;
    const probabilities=candidates.map((c,i)=>`<g role="button" tabindex="0" data-learn-open="probabilities" aria-label="${esc(token(c.text))}: ${pct(c.probability)}"><text class="prediction-token" x="1275" y="${123+i*29}">${esc(token(c.text).slice(0,14))}</text><rect class="prediction-track" x="1370" y="${112+i*29}" width="102" height="13" rx="2"/><rect class="prediction-bar${state.view==="pruned"?" pruned":""}" x="1370" y="${112+i*29}" width="${102*c.probability}" height="13" rx="2"/><text class="prediction-probability" x="1521" y="${123+i*29}" text-anchor="end">${pct(c.probability)}</text><title>Token ${c.id}: ${esc(c.text)} · ${pct(c.probability)}</title></g>`).join("");
    const topology=state.view==="original"?run.baseline_view:run.pruning_views?.[state.variant];
    const weights=topology?.layers?.[state.layer]?.mlp_weights?.find(w=>w.name.endsWith("c_fc.weight"));
    let weightsDiagram="";
    if(weights){const scale=absMax(weights.sample_values);weightsDiagram=`<g data-learn-open="mlp" role="button" tabindex="0"><text class="column-subtitle" x="863" y="${height-59}">MLP weight sample · masked tiles in coral</text>${weights.sample_values.map((value,i)=>`<rect x="${864+(i%24)*6}" y="${height-50+Math.floor(i/24)*5}" width="5" height="4" fill="${weights.sample?.[i]?"#d98e78":color(value,scale,tones.mlp)}"><title>${weights.sample?.[i]?"Masked to zero":"Retained weight"}: ${num(value)}</title></rect>`).join("")}</g>`;}
    return `<svg class="learn-network" viewBox="0 0 1560 ${height}" role="group" aria-label="Transformer Explainer-style left-to-right network: embedding, query key value, self-attention, MLP, remaining blocks, next-token probabilities">
      ${title("Embedding",22,"embeddings",`${p.hidden_size} dimensions / token`)}${title("Query · Key · Value",261,"qkv",`${p.head_count} attention heads`,p.layers.length)}${title("Self Attention",477,"attention",`Head ${state.head+1} · layer ${state.layer+1}`,p.layers.length)}${title("MLP",874,"mlp",`${p.hidden_size} → ${p.hidden_size*4} → ${p.hidden_size}`,p.layers.length)}${title("Other blocks",1096,"blocks","",p.layers.length)}${title("Linear + Softmax",1275,"probabilities",state.query===n-1?"Start of the continuation":`After token ${state.query}: ${token(p.tokens[state.query].text)}`)}
      ${state.layer?`<text class="column-subtitle" x="193" y="85">${state.layer} earlier blocks →</text>`:""}
      ${embeddedPaths}${tokenRows}
      <g class="${bypassed?"faded-computation":""}">
        <text class="query-label" x="282" y="91">Q</text><text class="key-label" x="316" y="91">K</text><text class="value-label" x="350" y="91">V</text>${qkv}${attentionPaths}
        <rect class="attention-back" x="${matrixX-1}" y="${matrixY-12}" width="${n*cell+27}" height="${n*cell+31}" rx="6"/><rect class="attention-back" x="${matrixX-7}" y="${matrixY-6}" width="${n*cell+27}" height="${n*cell+31}" rx="6"/><rect class="attention-stack" x="${matrixX-13}" y="${matrixY}" width="${n*cell+27}" height="${n*cell+31}" rx="6"/>
        ${matrixCells(p,data,matrixX-2,matrixY+12,cell)}
        <text class="column-subtitle" x="${matrixX-3}" y="${matrixY+n*cell+49}">${data.masked_heads.length?`${data.masked_heads.length}/${p.head_count} head outputs masked`:`${p.head_count} heads · select a cell`}</text>
        ${headOutputs}<text class="column-subtitle" x="752" y="91">Out</text><text class="column-subtitle" x="791" y="85">concat /</text><text class="column-subtitle" x="791" y="97">project</text>${mlp}
        ${masked?`<path class="pruned-route" d="M735,${mid-8} l16,16 M751,${mid-8} l-16,16"/><text class="mask-label" x="726" y="${mid+50}">zeroed</text>`:""}
      </g>
      ${output}${bypassed?`<path class="bypass-route" d="M243,${mid} V${height-79} H1059 V${mid}"/><text class="bypass-label" x="459" y="${height-85}">Block bypassed · effective output = input</text>`:""}
      ${route(1071,row(state.query),1121,mid,"",true)}${rest}${route(1192,mid,1241,mid,"",true)}<rect class="other-block" x="1232" y="${mid-22}" width="22" height="44" rx="3"/><text class="column-subtitle" x="1214" y="${mid+43}">projection</text>
      ${probabilities}<text class="column-subtitle" x="1275" y="${123+candidates.length*29}">Other tokens: ${pct(prediction.other_probability)}</text>
      <rect class="prediction-box" x="1275" y="${height-87}" width="246" height="46" rx="6"/><text class="prediction-answer" x="1398" y="${height-58}" text-anchor="middle">${esc(token(candidates[0].text))}</text>
      ${weightsDiagram}
    </svg>`;
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
      body+=`<svg class="learn-attention-matrix" viewBox="0 0 ${size} ${size}" aria-label="Measured attention matrix">${p.tokens.map((_,i)=>`<text x="${32+i*20}" y="14">${i}</text><text x="8" y="${36+i*20}">${i}</text>`).join("")}${matrixCells(p,data,26,25,20,state.matrix)}</svg>`;
      body+=row("Actual head contribution",data.head_output[state.head][state.query],"head_output",tones.out);
    }else if(state.drawer==="mlp"){
      heading="MLP and residual stream";body=`<p>Linear expansion → GELU → linear projection. The MLP output is added to the residual stream. ${data.masked_weights?`${data.masked_weights.toLocaleString()} of ${data.total_mlp_weights.toLocaleString()} MLP parameters are masked in this layer.`:"No MLP weight mask is applied in this view."} ${data.bypassed?"This block is bypassed: its internally computed values are discarded, and its effective output equals its input.":""}</p>`;
      body+=row("MLP output",data.mlp_output[state.query],"mlp_output",tones.mlp)+row("Block input",data.input[state.query],"input",tones.embedding)+row("Effective block output",data.output[state.query],"output",tones.out);
    }else if(state.drawer==="blocks"){
      heading="Transformer blocks";body='<p>GPT-2 repeats attention and MLP computations across twelve layers. Choose a layer above. Input embeddings are shared; inspecting a later layer does not skip the earlier layers in the actual model.</p>';
      body+=`<div class="learn-matrix-modes">${p.layers.map((l,i)=>`<button type="button" data-learn-layer="${i}" aria-pressed="${state.layer===i}">L${i+1}${l.bypassed?" · bypassed":l.masked_heads.length?" · heads masked":l.masked_weights?" · weights masked":""}</button>`).join("")}</div>`;
    }else{
      heading="Next-token probabilities";body=`<p>Final layer norm → vocabulary projection → softmax. Predictions after token ${state.query}: ${esc(token(p.tokens[state.query].text))}. The last input token predicts the start of the continuation. Earlier selections inspect earlier prefixes. Top-eight probabilities are not renormalized; other vocabulary tokens retain ${pct(p.predictions[state.query].other_probability)} of the probability mass.</p>`;
    }
    return `<section class="learn-inspector" aria-label="Expanded ${esc(heading)} details"><header><h3>${heading} · ${state.view} · layer ${state.layer+1}, head ${state.head+1}</h3><button type="button" data-learn-close>Close ×</button></header>${body}</section>`;
  }
  function completion(answer,label){
    const text=answer.text||"",snippet=text.slice(0,160),stop=answer.truncated?"context exhausted":answer.stop_reason==="eos"?"end of text":answer.stop_reason||"finished";
    return `<article><h3>${label}</h3><p class="learn-output-preview">${esc(snippet)}${text.length>160?" …":""}</p><details><summary>Full continuation · ${answer.tokens} tokens · ${esc(stop)}</summary><div>${esc(text)}</div></details></article>`;
  }
  function paint(){
    const scrollLeft=root().querySelector(".learn-canvas-scroll")?.scrollLeft||0;
    const run=state.run,current=pair();
    if(!run||["running","queued","cancelling"].includes(run.status)){root().innerHTML='<div class="learn-loading">Loading the prepared transformer…</div>';return;}
    if(!current){root().innerHTML=`<div class="learn-loading">${esc(run.baselines?.[0]?.pipeline_error||"Loading the measured original and pruned states…")}</div>`;return;}
    const {a,b,p}=current,preset=run.walkthrough_presets[state.variant];
    state.layer=Math.min(state.layer,p.layers.length-1);state.head=Math.min(state.head,p.head_count-1);
    if(state.query<0||state.query>=p.tokens.length)state.query=p.tokens.length-1;
    state.key=Math.min(state.key,p.tokens.length-1);
    const short={layers:"The last block's contribution is bypassed.",heads:"Three head outputs per layer are zeroed.",magnitude:"20% of the smallest MLP parameters are zeroed.",random:"About 20% of MLP parameters are zeroed at seeded random positions."}[preset.technique];
    root().innerHTML=`<div class="learn-toolbar"><div class="learn-state-switch" role="group" aria-label="Model state"><button type="button" data-model-state="original" aria-pressed="${state.view==="original"}">Original</button><button type="button" data-model-state="pruned" aria-pressed="${state.view==="pruned"}">Pruned</button></div>${indexControl("layer","Layer",p.layers.length)}${indexControl("head","Head",p.head_count)}<span class="learn-scroll-hint">Scroll Layer / Head to explore</span><span class="learn-preset-note"><strong>${state.view==="original"?"Original model shown. Pruned preset:":"Pruned model shown."}</strong> ${esc(short)}</span></div><div class="learn-canvas-scroll" tabindex="0" aria-label="Transformer network; scroll horizontally on small screens">${graph(a,b,p)}</div><div class="learn-diagram-caption"><span>Scroll Layer / Head or use the arrows above a block. Click a heading, token or attention cell for details.</span><span>Tile intensity = sample magnitude; hover for signed values. Routes and expansion shapes are architectural.</span></div>${inspector(a,b,p)}<section class="learn-output-comparison" aria-label="Original and pruned text output">${completion(run.baselines[0],"Original completion · preview")}${completion(run.comparisons[`${state.variant}:0`].answer,`Pruned completion · ${esc(preset.title)}`)}</section>`;
    root().querySelector(".learn-canvas-scroll").scrollLeft=scrollLeft;
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
    if(item.disabled||item.getAttribute("aria-disabled")==="true")document.getElementById(`learn-${item.dataset.learnAxis}`)?.focus({preventScroll:true});
    else item.focus({preventScroll:true});
  }
  function render(run,visible,question=0,variant=0,onSelect=null){
    root().hidden=!visible;if(!visible)return;
    const newRun=(run?.id??null)!==state.id;
    if(newRun){const preset=run?.walkthrough_presets?.[variant];state.id=run?.id??null;state.view="original";state.layer=preset?.inspection_layer??0;state.head=preset?.inspection_head??0;state.query=-1;state.key=0;state.drawer=null;state.frame="";}
    if(run&&variant!==state.variant){const preset=run.walkthrough_presets[variant];state.view="pruned";state.layer=preset.inspection_layer;state.head=preset.inspection_head;}
    state.run=run;state.variant=variant;state.select=onSelect;
    const frame=`${run?.id}:${run?.revision}:${variant}:${!!run?.baselines?.[0]?.pipeline}:${!!run?.comparisons?.[`${variant}:0`]?.answer?.pipeline}:${run?.baselines?.[0]?.pipeline_error||""}`;
    if(frame===state.frame)return;state.frame=frame;paint();
  }
  document.addEventListener("DOMContentLoaded",()=>{
    const activate=event=>{
      const item=event.target.closest("[data-model-state],[data-learn-open],[data-learn-token],[data-learn-cell],[data-learn-close],[data-learn-layer],[data-learn-matrix],[data-learn-step]");if(!item)return;
      if(event.type==="keydown"&&event.key!=="Enter"&&event.key!==" ")return;
      if(item.disabled||item.getAttribute("aria-disabled")==="true")return;
      event.preventDefault();let focus=null;
      if(item.dataset.learnStep!==undefined){if(!move(item.dataset.learnAxis,+item.dataset.learnStep))return;focus=`[data-learn-nav="${item.dataset.learnNav}"]`;}
      else if(item.dataset.modelState){state.view=item.dataset.modelState;focus=`[data-model-state="${state.view}"]`;}
      else if(item.hasAttribute("data-learn-close")){state.drawer=null;focus='[data-model-state="'+state.view+'"]';}
      else if(item.dataset.learnOpen){state.drawer=state.drawer===item.dataset.learnOpen?null:item.dataset.learnOpen;focus=`[data-learn-open="${item.dataset.learnOpen}"]`;}
      else if(item.dataset.learnToken!==undefined){state.query=+item.dataset.learnToken;state.key=Math.min(state.key,state.query);focus=`[data-learn-token="${state.query}"]`;}
      else if(item.dataset.learnCell){[state.query,state.key]=item.dataset.learnCell.split(",").map(Number);state.drawer="attention";}
      else if(item.dataset.learnLayer!==undefined){state.layer=+item.dataset.learnLayer;}
      else if(item.dataset.learnMatrix){state.matrix=item.dataset.learnMatrix;focus=`[data-learn-matrix="${state.matrix}"]`;}
      paint();restoreFocus(focus);
    };
    root().addEventListener("click",activate);root().addEventListener("keydown",activate);
    root().addEventListener("change",event=>{const id=event.target.id;if(id==="learn-layer")state.layer=+event.target.value;else if(id==="learn-head")state.head=+event.target.value;else return;paint();document.getElementById(id)?.focus({preventScroll:true});});
    let wheelAxis=null,wheelDelta=0,wheelTime=0,lastStep=-Infinity;
    root().addEventListener("wheel",event=>{
      const control=event.target.closest("[data-learn-scroll]");
      if(!control||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey||Math.abs(event.deltaX)>Math.abs(event.deltaY)||!event.deltaY)return;
      event.preventDefault();
      const axis=control.dataset.learnScroll,now=performance.now();
      if(axis!==wheelAxis||now-wheelTime>250||Math.sign(wheelDelta)!==Math.sign(event.deltaY))wheelDelta=0;
      wheelAxis=axis;wheelTime=now;
      wheelDelta+=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?window.innerHeight:1);
      if(Math.abs(wheelDelta)<24||now-lastStep<120)return;
      const delta=Math.sign(wheelDelta);wheelDelta=0;
      if(!move(axis,delta))return;
      lastStep=now;
      const active=document.activeElement,focus=active?.id?`#${active.id}`:active?.dataset.learnNav?`[data-learn-nav="${active.dataset.learnNav}"]`:null;
      paint();restoreFocus(focus);
    },{passive:false});
  });
  return {render};
})();
