// 1-hour Crunch mode
(function(){
  "use strict";
  const APP = window.__AI222, PLAN = window.CRUNCH || [];
  if (!APP || !PLAN.length) return;
  const {ALL, TOPICS, KINDS, typeset, store, save} = APP;
  const BY_ID = {}; ALL.forEach(c => BY_ID[c.id] = c);
  const TOTAL_MS = 60*60*1000;
  const normal = document.getElementById("normal"), box = document.getElementById("crunch");
  const topicVar = id => (id==="mix" ? "var(--accent)" : "var(--"+id+")");

  let cs = store.crunch || null;      // persisted crunch state
  let ticker = null;

  // ---------- time ----------
  function elapsed(){ return cs ? cs.accum + (cs.startTs ? Date.now()-cs.startTs : 0) : 0; }
  function fmt(ms){ const neg = ms<0; ms=Math.abs(ms); const m=Math.floor(ms/60000), s=Math.floor(ms%60000/1000);
    return (neg?"−":"")+m+":"+String(s).padStart(2,"0"); }
  function pause(){ if(cs && cs.startTs){ cs.accum += Date.now()-cs.startTs; cs.startTs=null; persist(); } }
  function resume(){ if(cs && !cs.startTs && cs.phase!=="done"){ cs.startTs=Date.now(); persist(); } }
  function persist(){ store.crunch = cs; save(); }

  // ---------- enter / exit ----------
  function enter(){
    document.body.classList.add("crunch-on");
    normal.hidden = true; box.hidden = false;
    render(); startTicker(); window.scrollTo({top:0});
  }
  function exit(){
    pause(); stopTicker();
    document.body.classList.remove("crunch-on");
    box.hidden = true; normal.hidden = false; APP.refresh();
  }
  document.getElementById("crunchBtn").onclick = () => document.body.classList.contains("crunch-on") ? exit() : enter();

  function startTicker(){ stopTicker(); ticker = setInterval(tick, 1000); tick(); }
  function stopTicker(){ if(ticker) clearInterval(ticker); ticker=null; }

  // ---------- state helpers ----------
  function newState(){
    return {block:0, phase:"sheet", queue:[], pos:0, revealed:false, accum:0, startTs:null, hideTimer:false, done:{}, got:{}};
  }
  function block(){ return PLAN[cs.block]; }
  function goBlock(i){
    cs.block = i; cs.phase = "sheet"; cs.queue = []; cs.pos = 0; cs.revealed = false; persist(); render(); window.scrollTo({top:0});
  }
  function startDrill(){
    cs.phase = "drill"; cs.queue = block().drill.filter(id => BY_ID[id]); cs.pos = 0; cs.revealed = false; persist(); render(); window.scrollTo({top:0});
  }
  function finishBlock(){
    cs.done[block().id] = true;
    if (cs.block < PLAN.length-1) goBlock(cs.block+1);
    else { cs.phase = "done"; pause(); persist(); render(); }
  }

  // ---------- render ----------
  function bar(){
    const b = block(), el = elapsed();
    const blockLeft = b.end*60000 - el, totalLeft = TOTAL_MS - el;
    const steps = PLAN.map((p,i) =>
      '<button class="step'+(cs.done[p.id]?' done':'')+'" data-i="'+i+'" style="--topic:'+topicVar(p.id)+'" aria-current="'+(i===cs.block)+'"><span class="dot"></span>'+p.title+' · '+p.mins+'m</button>').join("");
    return '<div class="cbar"><div class="cbar-top">'+
      (cs.hideTimer
        ? '<div class="clock"><span class="small">Timer hidden</span></div>'
        : '<div class="clock"><span class="big" id="cBlock">'+fmt(blockLeft)+'</span><span class="small">left in '+b.title+'</span><span class="small" id="cTotal">'+fmt(totalLeft)+' total</span></div>')+
      '<div class="row">'+
        '<button class="chip" id="cPause">'+(cs.startTs?'⏸ Pause':'▶ Resume')+'</button>'+
        '<button class="chip" id="cHide">'+(cs.hideTimer?'Show timer':'Hide timer')+'</button>'+
        '<button class="chip" id="cExit">✕ Exit</button>'+
      '</div></div>'+
      (cs.hideTimer?'':'<div class="tbar"><i id="cProg"></i></div>')+
      '<div class="steps">'+steps+'</div></div>';
  }

  function render(){
    if (!cs){ return renderIntro(); }
    if (cs.phase==="done") return renderDone();
    const b = block();
    let body = "";
    if (cs.phase==="sheet"){
      body = '<article class="sheet" style="--topic:'+topicVar(b.id)+'">'+
        '<h2>'+b.title+'</h2><p class="lead">Must-know sheet · read it in about '+Math.max(2,Math.round(b.mins*0.35))+' min, then drill '+b.drill.length+' cards.</p>'+
        b.sheet+
        '<div class="cta"><button class="btn primary" id="cDrill">Start drill: '+b.drill.length+' cards →</button></div></article>';
    } else {
      body = renderCard();
    }
    box.innerHTML = bar() + body;
    wire(); typeset(box); tick();
  }

  function renderCard(){
    const b = block();
    if (!cs.queue.length){
      return '<article class="sheet" style="--topic:'+topicVar(b.id)+'"><h2>'+b.title+' done ✓</h2>'+
        '<p>'+(cs.block<PLAN.length-1 ? 'Next up: <b>'+PLAN[cs.block+1].title+'</b>.' : 'That was the last block.')+'</p>'+
        '<div class="cta"><button class="btn primary" id="cNext">'+(cs.block<PLAN.length-1?'Next block →':'Finish')+'</button></div></article>';
    }
    const c = BY_ID[cs.queue[cs.pos]];
    return '<article class="card" style="--topic:var(--'+c.t+')">'+
      '<div class="meta"><span class="tag topic">'+TOPICS[c.t]+'</span><span class="tag">'+KINDS[c.k]+'</span>'+
      '<span class="count">'+cs.queue.length+' left in drill</span></div>'+
      '<h2 class="q">'+c.q+'</h2><div class="reveal-zone">'+
      (cs.revealed
        ? '<div class="answer"><p class="sec-h">Answer</p><div>'+c.a+'</div></div>'+
          '<div class="why"><p class="sec-h">Rationale</p><div>'+c.r+'</div></div>'+
          '<p class="src">Source: '+c.s+'</p>'+
          '<div class="rate"><button class="btn again" id="cAgain">↺ Again <kbd>1</kbd></button><button class="btn good" id="cGood">✓ Got it <kbd>2</kbd></button></div>'
        : '<button class="btn primary" id="cShow">Show answer <kbd>Space</kbd></button>')+
      '</div></article>'+
      '<div class="nav"><button class="btn" id="cSkip">Skip block →</button></div>';
  }

  function renderIntro(){
    const rows = PLAN.map(p => '<tr><td><span class="tag topic" style="--topic:'+topicVar(p.id)+'">'+p.title+'</span></td><td>Sheet + '+p.drill.length+' cards</td><td>'+p.mins+' min</td></tr>').join("");
    box.innerHTML = '<article class="sheet"><h2>⚡ 1-hour Crunch</h2>'+
      '<p class="lead">Everything high-yield across all 4 topics in 60 minutes.</p>'+
      '<p>Each block has a <b>must-know sheet</b> (formulas, key numbers, traps), followed by a <b>drill</b> of the highest-yield cards. Missed cards come back within the block. The timer shows pace only; it never cuts you off, and you can pause or hide it.</p>'+
      '<table class="plan"><tr><th>Block</th><th>What</th><th>Time</th></tr>'+rows+'<tr><th colspan="2">Total</th><th>60 min</th></tr></table>'+
      '<div class="cta"><button class="btn primary" id="cStart">Start the clock (60:00) →</button></div>'+
      '<div class="nav"><button class="btn" id="cBack">← Back to all cards</button></div></article>';
    document.getElementById("cStart").onclick = () => { cs = newState(); cs.startTs = Date.now(); persist(); render(); };
    document.getElementById("cBack").onclick = exit;
    typeset(box);
  }

  function renderDone(){
    const el = elapsed();
    const total = PLAN.reduce((n,p)=>n+p.drill.length,0), got = Object.keys(cs.got).length;
    box.innerHTML = '<article class="sheet"><h2>Crunch complete ✓</h2>'+
      '<p>Time used: <b>'+fmt(el)+'</b>. First-try "Got it": <b>'+got+' / '+total+'</b> drill cards.</p>'+
      '<p>Use any time left to reread the <b>trap</b> boxes on each sheet. Tap a block above to review its sheet again.</p>'+
      '<div class="cta"><button class="btn primary" id="cRestart">Restart crunch</button></div>'+
      '<div class="nav"><button class="btn" id="cSheets">Review sheets</button><button class="btn" id="cBack">← Back to all cards</button></div></article>';
    document.getElementById("cRestart").onclick = () => { cs = null; persist(); render(); };
    document.getElementById("cSheets").onclick = () => { cs.phase="sheet"; cs.block=0; persist(); render(); };
    document.getElementById("cBack").onclick = exit;
  }

  // ---------- actions ----------
  const $ = id => document.getElementById(id);
  function wire(){
    $("cPause").onclick = () => { cs.startTs ? pause() : resume(); render(); };
    $("cHide").onclick = () => { cs.hideTimer=!cs.hideTimer; persist(); render(); };
    $("cExit").onclick = exit;
    box.querySelectorAll(".step").forEach(b => b.onclick = () => goBlock(+b.dataset.i));
    if ($("cDrill")) $("cDrill").onclick = startDrill;
    if ($("cShow")) $("cShow").onclick = reveal;
    if ($("cAgain")) $("cAgain").onclick = again;
    if ($("cGood")) $("cGood").onclick = good;
    if ($("cNext")) $("cNext").onclick = finishBlock;
    if ($("cSkip")) $("cSkip").onclick = finishBlock;
  }
  function reveal(){ if(!cs.revealed){ cs.revealed=true; persist(); render(); } }
  function good(){
    const id = cs.queue[cs.pos];
    if (!(id in cs.got)) cs.got[id] = true;
    store.mastered[id] = true;
    cs.queue.splice(cs.pos,1); if (cs.pos>=cs.queue.length) cs.pos=0;
    cs.revealed=false; persist(); render(); scrollCard();
  }
  function again(){
    const id = cs.queue[cs.pos];
    if (!(id in cs.got)) cs.got[id] = false;
    cs.queue.splice(cs.pos,1);
    const at = Math.min(cs.pos+3, cs.queue.length); cs.queue.splice(at,0,id);
    cs.revealed=false; persist(); render(); scrollCard();
  }
  function scrollCard(){ const c = box.querySelector(".card,.sheet"); if(c) window.scrollTo({top: Math.max(0, c.offsetTop-140), behavior:"smooth"}); }

  function tick(){
    if (!cs || cs.phase==="done") return;
    const el = elapsed(), b = block();
    const bl = $("cBlock"), tt = $("cTotal"), pr = $("cProg");
    if (bl){ const left = b.end*60000-el; bl.textContent = fmt(left); bl.classList.toggle("over", left<0);
      bl.title = left<0 ? "Over time for this block: wrap up and move on" : ""; }
    if (tt){ tt.textContent = fmt(TOTAL_MS-el)+" total"; }
    if (pr){ pr.style.width = Math.min(100, 100*el/TOTAL_MS)+"%"; }
  }

  // ---------- keyboard ----------
  document.addEventListener("keydown", e => {
    if (!document.body.classList.contains("crunch-on") || !cs || cs.phase!=="drill" || e.target.matches("input,textarea") || e.metaKey||e.ctrlKey||e.altKey) return;
    if (e.key===" " || e.key==="Enter"){ e.preventDefault(); if(!cs.queue.length) finishBlock(); else if(!cs.revealed) reveal(); }
    else if (e.key==="1" && cs.revealed) again();
    else if (e.key==="2" && cs.revealed) good();
  });
  document.addEventListener("visibilitychange", () => { if (cs) persist(); });
})();
