"use strict";
const byId=id=>document.getElementById(id);
const escapeText=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const walkthroughState={config:null,token:"",run:null,variant:0,revision:-1,timer:null,pairs:new Map()};
const walking=run=>["queued","running","cancelling"].includes(run.status);
function walkthroughNotice(message){byId("walkthrough-notice").textContent=message;byId("walkthrough-notice").hidden=!message;}
async function walkthroughApi(path,data){
  const response=await fetch(path,{method:data===undefined?"GET":"POST",headers:{"Content-Type":"application/json","X-Lab-Token":walkthroughState.token},...(data===undefined?{}:{body:JSON.stringify(data)})});
  if(!response.ok){const body=await response.json().catch(()=>({}));const error=new Error(typeof body.detail==="string"?body.detail:JSON.stringify(body.detail||`Request failed (${response.status})`));error.status=response.status;throw error;}
  return response.json();
}
function renderPresetCards(){
  const labels={layers:"Layer bypass",heads:"Head masking",magnitude:"Magnitude · 20%",random:"Random · 20%"};
  byId("walkthrough-presets").innerHTML=walkthroughState.config.presets.map((preset,i)=>`<button class="preset-card" type="button" data-preset="${i}" aria-pressed="${walkthroughState.variant===i}" title="${escapeText(preset.title)}">${labels[preset.technique]}</button>`).join("");
}
function walkthroughSelect(question,variant){walkthroughState.variant=variant;renderPresetCards();renderWalkthroughPair();}
async function renderWalkthroughPair(){
  const run=walkthroughState.run,vi=walkthroughState.variant;
  const draw=()=>window.gpt2View.render(run,true,0,vi,walkthroughSelect);
  draw();if(!run||walking(run))return;
  const baseline=run.baselines[0],pruned=run.comparisons[`${vi}:0`]?.answer;
  if(!baseline?.pipeline_available||!pruned?.pipeline_available)return;
  if(baseline.pipeline&&pruned.pipeline)return;
  const key=`${run.id}:${run.revision}:${vi}`;
  if(!walkthroughState.pairs.has(key)){
    walkthroughState.pairs.set(key,walkthroughApi(`/api/runs/${run.id}/pipeline?question=0&variant=${vi}`));
    if(walkthroughState.pairs.size>4)walkthroughState.pairs.delete(walkthroughState.pairs.keys().next().value);
  }
  try{
    const pair=await walkthroughState.pairs.get(key);
    if(walkthroughState.run!==run||walkthroughState.variant!==vi)return;
    Object.values(run.baselines).forEach(answer=>delete answer.pipeline);Object.values(run.comparisons).forEach(cell=>delete cell.answer.pipeline);
    baseline.pipeline=pair.baseline;pruned.pipeline=pair.pruned;draw();
  }catch(error){walkthroughState.pairs.delete(key);baseline.pipeline_error=error.message;if(walkthroughState.run===run)draw();}
}
function renderWalkthrough(run){
  if(run.kind!=="walkthrough"||run.walkthrough_version!==walkthroughState.config.version||run.config.prompt!==walkthroughState.config.sentence){throw new Error("The prepared record does not match this sentence.");}
  walkthroughState.run=run;
  byId("walkthrough-progress").hidden=!walking(run);byId("walkthrough-status").textContent="PREPARING EXAMPLE";byId("walkthrough-stage").textContent=run.stage;
  byId("walkthrough-count").textContent=`${run.done} / ${run.total} model states`;
  byId("walkthrough-progress-bar").max=run.total;byId("walkthrough-progress-bar").value=run.done;
  renderWalkthroughPair();if(run.error)walkthroughNotice(run.error);
}
async function pollWalkthrough(id){
  try{
    const run=await walkthroughApi(`/api/runs/${id}`);
    if(run.revision!==walkthroughState.revision){walkthroughState.revision=run.revision;renderWalkthrough(run);}
    if(walking(run)){walkthroughState.timer=setTimeout(()=>pollWalkthrough(id),1000);return;}
    if(run.status==="completed")walkthroughNotice("");
    else walkthroughNotice(run.error||"The example could not finish preparing. Refresh the page to retry automatically.");
  }catch(error){walkthroughNotice(`Could not load the prepared example: ${error.message}. Retrying…`);walkthroughState.timer=setTimeout(()=>pollWalkthrough(id),3000);}
}
async function loadPreparedWalkthrough(){
  try{
    const result=await walkthroughApi("/api/walkthrough/ready",{});
    walkthroughNotice("");await pollWalkthrough(result.id);
  }catch(error){
    if(error.status===409){walkthroughNotice("The shared worker is busy with Qwen. This example will load automatically when it is available.");walkthroughState.timer=setTimeout(loadPreparedWalkthrough,5000);}
    else walkthroughNotice(`Could not prepare the example: ${error.message}`);
  }
}
byId("walkthrough-presets").addEventListener("click",event=>{const button=event.target.closest("[data-preset]");if(button)walkthroughSelect(0,+button.dataset.preset);});
(async()=>{
  try{
    const config=await walkthroughApi("/api/walkthrough/config");walkthroughState.config=config;walkthroughState.token=config.token;
    byId("fixed-sentence").textContent=config.sentence;renderPresetCards();window.gpt2View.render(null,true);
    await loadPreparedWalkthrough();
  }catch(error){walkthroughNotice(`Could not load the walkthrough: ${error.message}`);}
})();
