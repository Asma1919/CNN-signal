var cp=a=>JSON.parse(JSON.stringify(a)),$=i=>document.getElementById(i),f2=v=>+v.toFixed(2);
var M={sig:["Données","Signal","Échantillonnage","Valeurs","Convolution","ReLU","Pooling","Empilement","Flatten","Couche dense","Softmax","Entraînement","Prédiction","Code complet"]};
var D={sig:{ph:1,pw:2,pk:{"Dérivée":[[-1,0,1]],"Moyenne":[[1,1,1]],"Pic":[[-1,2,-1]]}}};
var SG={"Carré":t=>t>.3&&t<.7?9:0,"Sinus":t=>4.5+4.5*Math.sin(2*Math.PI*t),"Pic":t=>9*Math.exp(-Math.pow((t-.5)/.1,2)),"Bruité":t=>Math.max(0,Math.min(9,4.5+3*Math.sin(2*Math.PI*t)+1.5*Math.sin(t*60)))};
var S={},TM=null,cur="sig",CL={"Convolution":"c-conv","ReLU":"c-relu","Pooling":"c-pool","Empilement":"c-stack","Couche dense":"c-dense","Softmax":"c-soft","Prédiction":"c-pred","Entraînement":"c-train","Code complet":"c-code"};
function acc(m){m=m||"sig";var s=S[m],I0=s.I,ok=0,L=0,B=DSS;B.forEach(function(o){s.I=o.I;var p=sm(zf(m));ok+=p.indexOf(Math.max.apply(null,p))==o.y?1:0;L+=-Math.log(Math.max(1e-9,p[o.y]))});s.I=I0;return [ok,L/B.length]}
function sigI(){var s=S.sig;s.I=[Array.from({length:s.n},(_,k)=>Math.max(0,Math.min(9,Math.round(SG[s.pre]((k+.5)/s.n)))))]}
function conv(I,K){var O=[];for(var i=0;i<=I.length-K.length;i++){O.push([]);for(var j=0;j<=I[0].length-K[0].length;j++){var s=0;for(var a=0;a<K.length;a++)for(var b=0;b<K[0].length;b++)s+=I[i+a][j+b]*K[a][b];O[i].push(f2(s))}}return O}
function vals(A,i,j,h,w){var v=[];for(var a=0;a<h;a++)for(var b=0;b<w;b++)v.push(A[i+a][j+b]);return v}
function pool(A,ph,pw,t){var P=[];for(var i=0;i+ph<=A.length;i+=ph){P.push([]);for(var j=0;j+pw<=A[0].length;j+=pw){var v=vals(A,i,j,ph,pw);P[P.length-1].push(t=="max"?Math.max(...v):f2(v.reduce((x,y)=>x+y,0)/v.length))}}return P}
var gs=g=>{g=Math.max(0,Math.min(255,Math.round(g)));return `background:rgb(${g},${g},${g});color:${g>128?"#000":"#fff"}`};
var heat=(v,mx)=>`background:${v>=0?"rgba(31,111,178,":"rgba(217,130,43,"}${Math.min(1,Math.abs(v)/(mx||1))*.55})`;
var T=(m,A,o)=>"<table>"+A.map((r,i)=>"<tr>"+r.map((v,j)=>{var c=o.f?o.f(i,j):"",st=o.s?o.s(i,j,v):"";return `<td class="${c}${o.k?" k":""}" style="${st}"${o.k?` onclick="pk('${m}','${o.k}',${i},${j})"`:""}>${o.e?`<input type="number" step="any" value="${v}" onchange="ed('${m}','${o.e}',${i},${j},this.value)">`:v}</td>`}).join("")+"</tr>").join("")+"</table>";
var hd=(t,x)=>`<h2>${t}</h2><p>${x}</p>`;
function R(m){var s=S[m],nm=M[m][s.step],d=D[m],h='<div class="prog"><i style="width:'+Math.round((s.step+1)/M[m].length*100)+'%"></i></div><div class="stepno">'+PT(nm)+' · Étape '+(s.step+1)+' / '+M[m].length+'</div><div class="steps">'+M[m].map((n,i)=>`<button class="${i==s.step?"on":i<s.step?"done":""}" onclick="go('${m}',${i})">${i+1}. ${n}</button>`).join("")+`</div><div class="card ${CL[nm]||"c-info"}${s.fresh?" fade":""}">`+(NS.indexOf(nm)>=0?stat(m):"");s.fresh=0;
s.K=s.Ks[s.f];var I=s.I,K=s.K,A=conv(I,K),B=s.relu?A.map(r=>r.map(v=>Math.max(0,v))):A,P=pool(B,d.ph,d.pw,s.type);
s.c[0]=Math.min(s.c[0],A.length-1);s.c[1]=Math.min(s.c[1],A[0].length-1);s.p[0]=Math.min(s.p[0],P.length-1);s.p[1]=Math.min(s.p[1],P[0].length-1);
var n_=M[m].indexOf(nm)+1,ci=s.c[0],cj=s.c[1],pi=s.p[0],pj=s.p[1],kh=K.length,kw=K[0].length,mx=Math.max(1,...A.flat().map(Math.abs));
var win=(i,j)=>i>=ci&&i<ci+kh&&j>=cj&&j<cj+kw?"hl":"";
if(nm=="Données"){var ij=m=="img",nms=CLS[m];h+=hd(n_+". Données étiquetées","C’est le point de départ de l’apprentissage supervisé : chaque "+"signal"+" du jeu de données arrive avec son étiquette, donnée par un humain. En vrai projet, il y en a des milliers par classe.")+('<div class="dsg sg">'+dssHTML()+'</div><p>'+DSS.length+' signaux étiquetés : plusieurs exemples par classe (positions, hauteurs et phases différentes).</p>')+'<div class="learn"><h3>Le plan : 2 parties</h3><p><b>Partie 1 · Apprentissage :</b> données étiquetées, le CNN avec ses couches, puis l’entraînement qui produit les filtres, W et b.</p><p><b>Partie 2 · Test :</b> une nouvelle '+"entrée"+' passe dans le CNN déjà entraîné et donne la prédiction.</p></div><p>Pour suivre le calcul pas à pas, on prend un exemple : '+"un signal"+'.</p>'}
else if(nm=="Signal")h+=hd(n_+". Le signal","Un signal est une grandeur qui varie dans le temps (capteur, audio, ECG…). Choisissez un signal.")+'<canvas id="cv0" width="320" height="120" style="width:100%;max-width:320px"></canvas><div class="row">'+Object.keys(SG).map(n=>`<button class="${n==s.pre?"on":""}" onclick="setPre('${n}')">${n}</button>`).join("")+"</div>";
else if(nm=="Échantillonnage")h+=hd(n_+". Échantillonnage","Le capteur ne mesure pas en continu : il prend une mesure à intervalles réguliers. Chaque point est un échantillon.")+'<canvas id="cv0" width="320" height="120" style="width:100%;max-width:320px"></canvas>'+`<div class="row"><label>Nombre d’échantillons : <select onchange="setN('sig',this.value)">${[10,12,16].map(n=>`<option${n==s.n?" selected":""}>${n}</option>`).join("")}</select></label></div>`;
else if(nm=="Valeurs")h+=hd(n_+". Les valeurs numériques","Chaque échantillon devient un nombre (ici de 0 à 9). Le signal est maintenant une liste de valeurs : c’est l’entrée du CNN.")+'<div class="flow">'+T(m,I,{s:(i,j,v)=>heat(v,9)})+"</div>";
else if(nm=="Convolution"){var sum=0,terms=[];for(var a=0;a<kh;a++)for(var b=0;b<kw;b++){var x=I[ci+a][cj+b],w=K[a][b];terms.push(`(${x}×${w})`);sum+=x*w}sum=f2(sum);s.corr=terms.join(" + ")+" = "+sum;
h+=hd((M[m].indexOf(nm)+1)+". Convolution","Le filtre glisse sur l’entrée : à chaque position, on multiplie terme à terme puis on additionne. Touchez une case du résultat : calculez-la vous-même, entrez la valeur, puis vérifiez.")+'<div class="row"><button class="go" onclick="play(\''+m+'\',\'c\')">Animer le filtre</button>'+`<button onclick="rndK('${m}')">Nouveau filtre aléatoire</button>`+'</div><div class="flow"><div>'+T(m,I,{e:"I",f:win,s:(i,j,v)=>heat(v,9)})+'</div><span class="op">*</span><div>'+T(m,K,{e:"K",s:(i,j,v)=>heat(v,1)})+'</div><span class="op">=</span><div>'+T(m,A,{k:"c",f:(i,j)=>i==ci&&j==cj?"sel":"",s:(i,j,v)=>heat(v,mx)})+"</div></div>"
+`<p>Violet : valeur positive. Orange : valeur négative.</p><div class="calc">Case (${ci+1}, ${cj+1}) : multipliez chaque valeur de la zone orange par la valeur du filtre, puis additionnez.</div><div class="row"><span style="font-size:14px">À vous : entrez le résultat de cette case</span><input class="ans" id="${m}a" type="number" step="any"><button onclick="chk('${m}',${sum})">Vérifier et voir la correction</button></div><div id="${m}r" class="calc"></div>`+'<div class="learn"><p>Les valeurs du filtre sont <b>obtenues grâce à l’entraînement</b>, pas choisies à la main.</p><button class="go" onclick="go(\''+m+'\','+M[m].indexOf("Entraînement")+')">Voir l’entraînement →</button></div>'}
else if(nm=="ReLU"){var neg=A.flat().filter(function(v){return v<0}).length;h+=hd(n_+". ReLU","ReLU(x) = max(0, x) : une valeur positive reste identique, une valeur négative devient 0. Les cases entourées sont les valeurs négatives avant ReLU, et leur résultat après.")+`<div class="row"><label><input type="checkbox" ${s.relu?"checked":""} onchange="S.${m}.relu=this.checked;R('${m}')"> Activer ReLU</label></div><div class="flow"><div><p><b>Avant ReLU</b></p>`+T(m,A,{f:function(i,j){return A[i][j]<0?"hl":""},s:function(i,j,v){return heat(v,mx)}})+'</div><span class="op">→</span><div><p><b>Après ReLU</b></p>'+T(m,B,{f:function(i,j){return A[i][j]<0&&s.relu?"hl":""},s:function(i,j,v){return heat(v,mx)}})+`</div></div><div class="calc"><b>${neg}</b> valeur(s) négative(s) mise(s) à 0.</div><div class="row"><span style="font-size:14px">Essayez : x =</span><input class="ans" id="rx" type="number" value="-3" oninput="rl()"><span>→ ReLU(x) = <b id="ro">0</b></span></div>`}
else if(nm=="Pooling"){var pv=vals(B,pi*d.ph,pj*d.pw,d.ph,d.pw);
h+=hd((M[m].indexOf(nm)+1)+". Pooling "+d.ph+"×"+d.pw,"On garde le maximum de chaque zone (max pooling) : la taille diminue, mais l’information la plus forte est conservée. Les 0 viennent de ReLU : dès qu’une valeur positive est dans la zone, c’est elle qui est choisie, pas un 0. Touchez une case du résultat.")+`<div class="row"><button class="go" onclick="play('${m}','p')">Animer le pooling</button></div><div class="flow"><div>`+T(m,B,{f:(i,j)=>i>=pi*d.ph&&i<pi*d.ph+d.ph&&j>=pj*d.pw&&j<pj*d.pw+d.pw?(s.type=="max"&&B[i][j]==Math.max.apply(null,pv)?"hl2 mxc":"hl2"):"",s:(i,j,v)=>heat(v,mx)})+'</div><span class="op">→</span><div>'+T(m,P,{k:"p",f:(i,j)=>i==pi&&j==pj?"sel":"",s:(i,j,v)=>heat(v,mx)})+`</div></div><div class="calc">Zone verte : ${s.type=="max"?"max":"moyenne"}(${pv.join(", ")}) = <b>${P[pi][pj]}</b><br>Taille : ${A.length}×${A[0].length} → ${P.length}×${P[0].length}</div>`}
else if(nm=="Empilement"){var st=stk(m);h+=hd(n_+". Empilement des cartes","Le bloc Convolution, ReLU et pooling que vous venez de voir se répète avec d’autres filtres . Chaque filtre donne sa propre carte, et on empile ces cartes. C’est la même opération avec plusieurs filtres, pas une nouvelle opération.")+'<div class="flow">'+st.L.map(function(o){var q=Math.max(1,...o.P.flat().map(Math.abs));return "<div><p><b>"+o.n+"</b></p>"+T(m,o.K,{s:function(i,j,v){return heat(v,1)}})+"<p>après convolution, ReLU, pooling</p>"+T(m,o.P,{s:function(i,j,v){return heat(v,q)}})+"</div>"}).join("")+'</div><div class="calc">'+st.L.length+" filtres → "+st.L.length+" cartes de "+st.L[0].P.length+"×"+st.L[0].P[0].length+"</div><p>Ce bloc est répété à chaque itération de l’entraînement, et tous les filtres sont corrigés.</p>"}
else if(nm=="Flatten"){var st=stk(m),qm=Math.max(1,...st.vec.map(Math.abs));h+=hd(n_+". Flatten","On met toutes les cartes bout à bout pour former un seul vecteur 1D. C’est l’entrée de la couche dense.")+'<div class="flow">'+T(m,[st.vec],{s:function(i,j,v){return heat(v,qm)}})+'</div><div class="calc">'+st.L.length+" cartes × "+st.L[0].P.flat().length+" valeurs = <b>"+st.vec.length+"</b> valeurs</div>"}
else if(nm=="Couche dense"){var x=stk(m).vec,N=x.length,W=getW(m),qw=Math.max(.001,...W.flat().map(Math.abs)),b=s.b.map(function(r){return r[0]}),z=zf(m),cl=(s.zc||[0])[0],names=CLS[m],tm=[],qx=Math.max(1,...x.map(Math.abs));for(var i2=0;i2<3;i2++)tm.push("("+W[cl][i2]+"×"+x[i2]+")");
h+=hd(n_+". Couche dense (entièrement connectée)","Le vecteur x est combiné avec des poids W et un biais b : z = Wᵀx + b. Chaque classe a sa ligne de poids.")+"<p><b>Entrée x</b> ("+N+" valeurs)</p><div class=\"flow\">"+T(m,[x],{s:function(i,j,v){return heat(v,qx)}})+"</div>"
+'<div class="flow"><div><p><b>Poids W</b> (3×'+N+')</p>'+T(m,W.map(function(w,j){return [names[j]].concat(w.slice(0,4),["…"])}),{s:function(i,j,v){return typeof v=="number"?heat(v,qw):""}})+'</div><span class="op">+</span><div><p><b>Biais b</b></p>'+T(m,s.b,{e:"b"})+'</div><span class="op">=</span><div><p><b>Scores z</b></p>'+T(m,z.map(function(v,j){return [names[j],v]}),{k:"zc",f:function(i,j){return i==cl?"sel":""}})+"</div></div>"
+'<div class="calc">'+names[cl]+" : z = "+tm.join(" + ")+" + … + "+b[cl]+" = <b>"+z[cl]+"</b></div><p>Touchez une ligne de z pour voir son calcul. Modifiez b pour voir l’effet.</p>"+wb()}
else if(nm=="Softmax"){var z=zf(m),p=sm(z),names=CLS[m];h+=hd(n_+". Softmax","Les scores z peuvent avoir n’importe quelle valeur. Softmax les transforme en probabilités entre 0 et 1 dont la somme vaut 1 : p_i = e^(z_i) / Σ e^(z_j).")+'<table class="tt"><tr><th>Classe</th><th>Score z</th><th>e^z</th><th>Probabilité</th></tr>'+z.map(function(v,j){return "<tr><td>"+names[j]+"</td><td>"+v+"</td><td>"+f2(Math.exp(v))+'</td><td><div class="bar" style="width:'+Math.max(2,p[j]*100)+'%"></div> <b>'+f2(p[j])+"</b></td></tr>"}).join("")+'</table><div class="calc">Somme des probabilités = <b>'+f2(p.reduce(function(a,c){return a+c},0))+"</b></div>"}
else if(nm=="Entraînement"){var ac=acc("sig"),N=DSS.length,ok=ac[0],done=ac[1]<.05||s.tr>=300;
h+=hd(n_+". Entraînement","Ici le réseau apprend. On lui montre les "+N+" signaux avec leur bonne réponse (carré, sinus ou pic). Il essaie, se trompe, puis corrige ses réglages (filtres, W et b) pour se tromper moins.")
+'<div class="learn"><h3>Comment ça marche</h3><p><b>1 itération</b> = le réseau regarde les '+N+' signaux une fois et se corrige.</p><p><b>Pourquoi répéter ?</b> Une seule itération corrige très peu. Après des dizaines d’itérations, il se trompe de moins en moins.</p></div>'
+'<div class="dsg sg">'+dssHTML()+'</div>'
+'<div class="row"><button class="go" onclick="auto(\'sig\')">▶ Entraîner jusqu’au bout</button><button onclick="trn(\'sig\',1)">+1 itération</button><button onclick="rst(\'sig\')">Recommencer</button></div>'
+'<table class="tt"><tr><th>Itérations faites</th><td><b>'+s.tr+'</b></td><td>fois où le réseau a vu les '+N+' signaux</td></tr><tr><th>Signaux reconnus</th><td><b>'+ok+' sur '+N+'</b></td><td>bonne réponse trouvée</td></tr><tr><th>Erreur moyenne</th><td><b>'+f2(ac[1])+'</b></td><td>0 = parfait, plus c’est petit mieux c’est</td></tr></table>'
+curve(s.h||[],'Itérations','itération')+'<div class="learn"><h3>C’est quoi « l’erreur » (loss) ?</h3><p>On regarde la probabilité que le réseau donne à la <b>bonne</b> réponse. Erreur = −ln(probabilité).</p><table class="tt"><tr><th>Probabilité donnée à la bonne classe</th><th>Erreur</th></tr><tr><td>100 %</td><td>0 (parfait)</td></tr><tr><td>50 %</td><td>0,69</td></tr><tr><td>33 % (hasard, 3 classes)</td><td>1,10</td></tr><tr><td>10 %</td><td>2,30</td></tr></table><p>« Erreur moyenne » = la moyenne sur les '+N+' signaux.</p></div>'
+(done?'<div class="stat ok2">Entraînement terminé. <button class="go" onclick="go(\''+m+'\','+(s.step+1)+')">Passer au test →</button></div>':'')}
else if(nm=="Prédiction"){var I0=s.I,hasT=!s.dup;if(hasT)s.I=s.T;var z=hasT?zf(m):[0,0,0];s.I=I0;var p=sm(z),names=CLS[m],kk=p.indexOf(Math.max.apply(null,p)),rs=hasT?'<div class="res">'+names[kk]+'<small>'+f2(p[kk]*100)+' %</small></div>':'<span class="chip">Signal déjà vu à l’entraînement</span>';
h+=hd(n_+". Test : prédiction","Le réseau est déjà entraîné. Tu entres un signal, sans étiquette, et il répond par la classe qui a la plus grande probabilité. Touche le graphique ou modifie les valeurs (0 à 9) pour fabriquer ton signal : il doit être différent de ceux de l’entraînement.")
+'<canvas id="cv0" width="320" height="120" style="width:100%;max-width:320px;cursor:pointer" onclick="clkT(event,this)"></canvas>'
+'<div class="row">'+s.T[0].map(function(v,k){return '<input class="ans" style="width:50px" type="number" min="0" max="9" step="1" value="'+v+'" onchange="setTv('+k+',this.value)">'}).join("")+'</div>'
+'<div class="row"><span style="font-size:14px">Exemples inédits :</span><button onclick="presT(\'Carré\')">Carré</button><button onclick="presT(\'Sinus\')">Sinus</button><button onclick="presT(\'Pic\')">Pic</button><button onclick="presT(\'\')">Effacer</button></div>'
+(s.dup?'<div class="stat" style="background:rgba(176,42,42,.12);color:var(--ko)">Ce signal est identique à un signal d’entraînement. Modifie-le pour faire un vrai test.</div>':'')
+'<div class="flow"><span class="chip dk">CNN entraîné</span><span class="op">→</span>'+rs+'</div>'
+(hasT?'<table class="tt">'+p.map(function(v,j){return '<tr class="'+(j==kk?"win":"")+'"><td>'+names[j]+'</td><td><div class="bar" style="width:'+Math.max(2,v*100)+'%"></div> <b>'+f2(v)+"</b></td></tr>"}).join("")+'</table>':'')
+'<div class="learn"><h3>Pourquoi le réseau répond'+(hasT?' « '+names[kk]+' »':' ainsi')+' ?</h3><p>Parce qu’il a appris avant, pendant la Partie 1. Dans un vrai projet, on l’entraîne sur un grand jeu de signaux étiquetés par l’humain, puis on lui envoie un nouveau signal jamais vu : il prédit grâce aux filtres, W et b obtenus à l’entraînement.</p><button class="go" onclick="go(\''+m+'\','+M[m].indexOf("Entraînement")+')">Voir l’entraînement →</button></div>'}
else if(nm=="Code complet"){h+=hd(n_+". Code complet (Keras)","Voici le programme qui réalise tout ce que vous venez de voir. Vous écrivez l’architecture et lancez model.fit : le CNN apprend alors les filtres, W et b.")+'<div class="row"><button class="go" onclick="cpy(this)">Copier le code</button></div><pre>'+esc(FULL[m])+"</pre>"}
h+=cd(m,nm);h+=`</div><div class="nav"><div><button ${s.step?"":"disabled"} onclick="go('${m}',${s.step-1})">← Précédent</button><button class="on" ${s.step<M[m].length-1?"":"disabled"} onclick="go('${m}',${s.step+1})">Suivant →</button></div></div>`;
$(m).innerHTML=h;paint(m)}
function paint(m){var s=S[m],nm=M[m][s.step];paintSig(s,nm);
if(m=="sig"&&(nm=="Signal"||nm=="Échantillonnage")){var c=$("cv0").getContext("2d"),W=320,H=120,Y=v=>H-10-v*10,f=SG[s.pre];c.clearRect(0,0,W,H);c.strokeStyle="#888";c.lineWidth=1;c.beginPath();c.moveTo(0,H-10);c.lineTo(W,H-10);c.stroke();
c.strokeStyle="#7F77DD";c.lineWidth=2;c.beginPath();for(var x=0;x<=W;x++){var y=Y(f(x/W));x?c.lineTo(x,y):c.moveTo(x,y)}c.stroke();
if(nm=="Échantillonnage"){c.strokeStyle="#D85A30";c.fillStyle="#D85A30";c.lineWidth=1.5;for(var k=0;k<s.n;k++){var xk=(k+.5)/s.n*W,yk=Y(s.I[0][k]);c.beginPath();c.moveTo(xk,H-10);c.lineTo(xk,yk);c.stroke();c.beginPath();c.arc(xk,yk,3.5,0,7);c.fill()}}}}
function go(m,i){clearInterval(TM);var s=S[m];s.step=i;s.fresh=1;if(M[m][i]=="Pooling"){s.K=s.Ks[s.f];var o=fwd(m).F[s.f],dd=D[m],bz=-1e9;o.P.forEach(function(r,a){r.forEach(function(v,b){var z=vals(o.B,a*dd.ph,b*dd.pw,dd.ph,dd.pw),sc=(z.some(function(t){return t<=0})&&v>0?1000:0)+v;if(sc>bz){bz=sc;s.p=[a,b]}})})}R(m);window.scrollTo({top:0,behavior:"smooth"})}
function ed(m,k,i,j,v){S[m][k][i][j]=Number(v)||0;R(m)}
function pk(m,k,i,j){S[m][k]=[i,j];R(m)}
function setN(m,v){S[m].n=+v;S[m].W=null;sigI(),sigSet(),tvInit();R(m)}
function setPre(n){S.sig.pre=n;S.sig.W=null;sigI();R("sig")}
function chk(m,sum){var v=$(m+"a").value,e=$(m+"r");if(v===""){e.className="calc ko";e.textContent="Entrez un nombre d’abord.";return}
var okk=Math.abs(Number(v)-sum)<.015;e.className="calc "+(okk?"ok":"ko");e.innerHTML="<b>"+(okk?"Correct.":"Pas tout à fait.")+"</b> Vous avez entré "+v+".<br>Méthode : on multiplie chaque valeur de la zone par la valeur du filtre, puis on additionne tous les produits.<br>Correction : "+S[m].corr}
function play(m,k){clearInterval(TM);var s=S[m],d=D[m],A=conv(s.I,s.K),cols=k=="c"?A[0].length:Math.floor(A[0].length/d.pw),tot=k=="c"?A.length*cols:Math.floor(A.length/d.ph)*cols,i=0;
TM=setInterval(function(){if(i>=tot){clearInterval(TM);return}s[k]=[Math.floor(i/cols),i%cols];i++;R(m)},450)}
function loss(m){var s=S[m],t=conv(s.I,Object.values(D[m].pk)[0]),o=conv(s.I,s.K),e=0,c=0;o.forEach((r,i)=>r.forEach((v,j)=>{e+=Math.pow(v-t[i][j],2);c++}));return e/c}
var lb=m=>`<div class="learn"><h3 style="margin:0 0 4px;font-size:16px;font-weight:500">Comment le filtre est-il choisi ?</h3><p>Il n’est pas choisi à la main. Le concepteur fixe la taille et le nombre de filtres. Leurs valeurs sont d’abord aléatoires, puis apprises pendant l’entraînement : le réseau les corrige pour réduire l’erreur (descente de gradient).</p><p>Essayez : la cible est le filtre « ${Object.keys(D[m].pk)[0]} ». Partez d’un filtre aléatoire et regardez l’erreur baisser.</p><div class="row"><button onclick="rndK('${m}')">Filtre aléatoire</button><button class="go" onclick="learn('${m}',1)">Apprendre ×1</button><button class="go" onclick="learn('${m}',100)">Apprendre ×100</button></div><div class="calc">Erreur actuelle : <b>${f2(loss(m))}</b></div></div>`;
function learn(m,n){var s=S[m],Kt=Object.values(D[m].pk)[0],I=s.I,kh=s.K.length,kw=s.K[0].length,mx2=Math.max(.01,...I.flat().map(v=>v*v)),lr=.4/(kh*kw*mx2);
for(var it=0;it<n;it++){var o=conv(I,s.K),t=conv(I,Kt),G=s.K.map(r=>r.map(()=>0)),N=o.length*o[0].length;
for(var i=0;i<o.length;i++)for(var j=0;j<o[0].length;j++){var e=o[i][j]-t[i][j];for(var a=0;a<kh;a++)for(var b=0;b<kw;b++)G[a][b]+=2*e*I[i+a][j+b]/N}
s.K=s.K.map((r,a)=>r.map((v,b)=>v-lr*G[a][b]))}
s.K=s.K.map(r=>r.map(v=>+v.toFixed(3)));R(m)}
document.addEventListener("keydown",function(e){if(e.target.tagName=="INPUT"||e.target.tagName=="SELECT")return;var s=S[cur];if(e.key=="ArrowRight"&&s.step<M[cur].length-1)go(cur,s.step+1);if(e.key=="ArrowLeft"&&s.step>0)go(cur,s.step-1)});

var DSS=[];
function sigSet(){var n=S.sig.n,Q=function(f){return [Array.from({length:n},function(_,k){return Math.max(0,Math.min(9,Math.round(f((k+.5)/n))))})]},
C=[[.3,.7,9],[.25,.65,8],[.35,.75,9],[.2,.6,7],[.4,.8,8],[.3,.6,9],[.25,.7,8],[.35,.7,9]],Sn=[[0,4.5],[.05,4],[.95,4.5],[.1,3.5],[.9,4],[.15,4.5],[.85,3.5],[.97,4.5]],P=[[.5,9],[.4,8],[.6,9],[.45,7],[.55,9],[.35,8],[.65,9],[.5,7]];DSS=[];
for(var i=0;i<8;i++){var c=C[i],a=Sn[i],q=P[i];
DSS.push({I:Q(function(t){return t>c[0]&&t<c[1]?c[2]:0}),y:0});
DSS.push({I:Q(function(t){return 4.5+a[1]*Math.sin(2*Math.PI*(t+a[0]))}),y:1});
DSS.push({I:Q(function(t){return q[1]*Math.exp(-Math.pow((t-q[0])/.1,2))}),y:2})}}
function dssHTML(){return DSS.map(function(o,i){return '<figure class="th"><canvas id="ds'+i+'" width="104" height="46"></canvas><figcaption>'+CLS.sig[o.y]+'</figcaption></figure>'}).join("")}
function thumb(c,v){var W=104,H=46,n=v.length,Y=function(y){return H-5-y*(H-10)/9};c.fillStyle="#fff";c.fillRect(0,0,W,H);c.strokeStyle="#bbb";c.lineWidth=1;c.beginPath();c.moveTo(0,H-4.5);c.lineTo(W,H-4.5);c.stroke();c.strokeStyle="#7F77DD";c.lineWidth=2;c.beginPath();v.forEach(function(y,k){var x=(k+.5)/n*W;k?c.lineTo(x,Y(y)):c.moveTo(x,Y(y))});c.stroke();c.fillStyle="#D85A30";v.forEach(function(y,k){c.beginPath();c.arc((k+.5)/n*W,Y(y),2,0,7);c.fill()})}
function stems(c,v,W,H){var n=v.length,Y=function(y){return H-10-y*10};c.clearRect(0,0,W,H);c.strokeStyle="#888";c.lineWidth=1;c.beginPath();c.moveTo(0,H-10);c.lineTo(W,H-10);c.stroke();c.strokeStyle="#D85A30";c.fillStyle="#D85A30";c.lineWidth=1.5;v.forEach(function(y,k){var x=(k+.5)/n*W;c.beginPath();c.moveTo(x,H-10);c.lineTo(x,Y(y));c.stroke();c.beginPath();c.arc(x,Y(y),3.5,0,7);c.fill()})}
function paintSig(s,nm){if(nm=="Données"||nm=="Entraînement")DSS.forEach(function(o,i){var e=$("ds"+i);if(e)thumb(e.getContext("2d"),o.I[0])});
if(nm=="Prédiction"){var e=$("cv0");if(e)stems(e.getContext("2d"),s.T[0],320,120)}}
function mkT(nm){var n=S.sig.n,f={"Carré":function(t){return t>.38&&t<.78?9:0},"Sinus":function(t){return 4.5+4.5*Math.sin(2*Math.PI*(t+.08))},"Pic":function(t){return 9*Math.exp(-Math.pow((t-.58)/.1,2))}}[nm];return [Array.from({length:n},function(_,k){return f?Math.max(0,Math.min(9,Math.round(f((k+.5)/n)))):0})]}
function sigDup(){var s=S.sig;s.dup=DSS.some(function(o){return o.I[0].join()==s.T[0].join()})?1:0}
function tvInit(){S.sig.T=mkT("Carré");sigDup()}
function presT(nm){S.sig.T=mkT(nm);sigDup();R("sig")}
function setTv(k,v){S.sig.T[0][k]=Math.max(0,Math.min(9,Math.round(Number(v)||0)));sigDup();R("sig")}
function clkT(e,c){var r=c.getBoundingClientRect(),n=S.sig.n,x=(e.clientX-r.left)/r.width*320,y=(e.clientY-r.top)/r.height*120,k=Math.min(n-1,Math.max(0,Math.floor(x/320*n)));S.sig.T[0][k]=Math.max(0,Math.min(9,Math.round((110-y)/10)));sigDup();R("sig")}
var CLS={sig:["Carré","Sinus","Pic"]};
function stk(m){var f=fwd(m);return {L:f.F.map(function(o,i){return {n:"Filtre "+(i+1),P:o.P,K:o.K}}),vec:f.vec}}
function Wm(m,N){var sc=.08;return [0,1,2].map(function(j){return Array.from({length:N},function(_,i){return +(sc*Math.sin((i+1)*(j+2)*1.7+j)).toFixed(3)})})}
function zf(m){var x=stk(m).vec,W=getW(m),b=S[m].b;return W.map(function(w,j){return f2(w.reduce(function(a,wi,i){return a+wi*x[i]},0)+b[j][0])})}
function sm(z){var mx=Math.max.apply(null,z),e=z.map(function(v){return Math.exp(v-mx)}),t=e.reduce(function(a,c){return a+c},0);return e.map(function(v){return v/t})}
var NS=["Convolution","ReLU","Pooling","Empilement","Flatten","Couche dense","Softmax"];
function labOf(m){var i=CLS.sig.indexOf(S[m].pre);return i<0?1:i}
function rk(m){var d=D[m],k=d.pk[Object.keys(d.pk)[0]];return [0,1,2].map(function(){var a=k.map(function(r){return r.map(function(){return Math.random()-.5})}),fl=a.flat(),mu=fl.reduce(function(x,y){return x+y},0)/fl.length;return a.map(function(r){return r.map(function(v){return f2(v-mu)})})})}
function Wi(m,N){var sc=.006;return [0,1,2].map(function(j){return Array.from({length:N},function(_,i){return +(sc*Math.sin((i+1)*(j+2)*1.7+j)).toFixed(4)})})}
function fwd(m){var s=S[m],d=D[m],F=s.Ks.map(function(K){var A=conv(s.I,K),B=s.relu?A.map(function(r){return r.map(function(v){return Math.max(0,v)})}):A;return {K:K,A:A,B:B,P:pool(B,d.ph,d.pw,s.type)}});return {F:F,vec:[].concat.apply([],F.map(function(o){return o.P.flat()}))}}
function getW(m){var s=S[m],N=fwd(m).vec.length;if(!s.W||s.W[0].length!=N){s.W=Wi(m,N);s.b=[[0],[0],[0]];s.tr=0;s.h=[]}return s.W}
function train(m,n){var s=S[m],d=D[m];getW(m);var y,I,ph=d.ph,pw=d.pw,m2,I0=s.I,B=DSS;
for(var t=0;t<n*B.length;t++){var sp=B[t%B.length];s.I=sp.I;y=sp.y;I=s.I;m2=I.flat().reduce(function(a,v){return a+v*v},0)/I.flat().length;var f=fwd(m),x=f.vec,z=s.W.map(function(w,j){return w.reduce(function(a,wi,i){return a+wi*x[i]},0)+s.b[j][0]}),p=sm(z),dz=p.map(function(v,j){return v-(j==y?1:0)}),
dx=x.map(function(_,i){return s.W.reduce(function(a,w,j){return a+w[i]*dz[j]},0)}),lrW=.5/(1+x.reduce(function(a,v){return a+v*v},0)),lrK=LK*.2/(1+4*m2),off=0;
LS+=-Math.log(Math.max(1e-9,p[y]));if(t%B.length==B.length-1){s.h.push(LS/B.length);LS=0}
f.F.forEach(function(o){var dA=o.A.map(function(r){return r.map(function(){return 0})});
for(var pi=0;pi<o.P.length;pi++)for(var pj=0;pj<o.P[0].length;pj++){var g=dx[off++],cs=[],bi=null;for(var a=0;a<ph;a++)for(var b=0;b<pw;b++){var c=[pi*ph+a,pj*pw+b];cs.push(c);if(!bi||o.B[c[0]][c[1]]>o.B[bi[0]][bi[1]])bi=c}
if(s.type=="max")dA[bi[0]][bi[1]]+=g;else cs.forEach(function(c){dA[c[0]][c[1]]+=g/cs.length})}
if(s.relu)dA=dA.map(function(r,i){return r.map(function(v,j){return o.A[i][j]>0?v:0})});
var dK=o.K.map(function(r){return r.map(function(){return 0})});
for(var i=0;i<dA.length;i++)for(var j=0;j<dA[0].length;j++)for(var a=0;a<o.K.length;a++)for(var b=0;b<o.K[0].length;b++)dK[a][b]+=dA[i][j]*I[i+a][j+b];
o.K.forEach(function(r,a){r.forEach(function(v,b){r[b]=v-lrK*dK[a][b]})})});
s.W.forEach(function(w,j){w.forEach(function(wi,i){w[i]=wi-lrW*dz[j]*x[i]})});s.b.forEach(function(r,j){r[0]-=lrW*dz[j]})}
s.I=I0;s.tr+=n;if(s.h.length>300)s.h=s.h.slice(-300);
s.Ks=s.Ks.map(function(K){return K.map(function(r){return r.map(function(v){return +v.toFixed(3)})})});s.W=s.W.map(function(r){return r.map(function(v){return +v.toFixed(4)})});s.b=s.b.map(function(r){return [+r[0].toFixed(3)]})}
var LS=0;function isDone(m){return acc("sig")[1]<.05||S[m].tr>=300}
function auto(m){clearInterval(TM);TM=setInterval(function(){train(m,3);R(m);if(isDone(m))clearInterval(TM)},100)}
var LK=1;
function rst(m){var s=S[m];s.Ks=rk(m);s.W=null;s.tr=0;s.h=[];getW(m);R(m)}
function trn(m,n){train(m,n);R(m)}
function stat(m){return ""}
function curve(H,u){if(H.length<2)return '<p>Appuie sur « Entraîner » : la courbe de l’erreur apparaît ici.</p>';var mx=Math.max.apply(null,H)||1,n=H.length,pts=H.map(function(v,i){return (36+i/(n-1)*252).toFixed(1)+","+(120-v/mx*100).toFixed(1)}).join(" ");return '<h3>Courbe de l’erreur</h3><svg viewBox="0 0 300 160" class="cv"><line x1="36" y1="20" x2="36" y2="120" stroke="#000"/><line x1="36" y1="120" x2="290" y2="120" stroke="#000"/><polyline points="'+pts+'" fill="none" stroke="#c2410c" stroke-width="2.5"/><text x="40" y="14" font-size="10" fill="#000">Erreur ↑</text><text x="2" y="30" font-size="10" fill="#000">'+f2(mx)+'</text><text x="24" y="123" font-size="10" fill="#000">0</text><text x="36" y="136" font-size="10" fill="#000">1</text><text x="290" y="136" font-size="10" text-anchor="end" fill="#000">'+n+'</text><text x="163" y="152" font-size="11" text-anchor="middle" fill="#000">'+u+' →</text></svg><p>Chaque point = l’erreur moyenne après une itération. <b>Si la ligne descend, le réseau apprend.</b> Début : <b>'+f2(H[0])+'</b> · maintenant : <b>'+f2(H[n-1])+'</b>.</p>'}
function wb(){var m=cur,s=S[m];return '<div class="learn"><h3>D’où viennent W et b ?</h3><p>Les valeurs de <b>W</b> et de <b>b</b> ne sont pas choisies à la main : elles sont <b>obtenues grâce à l’entraînement</b>. À chaque itération, le réseau les corrige pour réduire l’erreur'+(s.tr?' ('+s.tr+' itérations déjà effectuées)':'')+'.</p><button class="go" onclick="go(\''+m+'\','+M[m].indexOf("Entraînement")+')">Voir l’entraînement →</button></div>'}
function PT(nm){return nm=="Prédiction"?"Partie 2 · Test":nm=="Code complet"?"Code":"Partie 1 · Apprentissage"}
function esc(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function rl(){var v=Number($("rx").value)||0;$("ro").textContent=Math.max(0,v)}
function showc(m){var e=$(m+"r");e.className="ok";e.textContent="Correction : "+S[m].corr}
function cpy(b){var t=b.parentNode.nextSibling.textContent;try{navigator.clipboard.writeText(t);b.textContent="Copié"}catch(e){b.textContent="Sélectionnez le code puis copiez"}}
var CK={
"Signal":`t = np.linspace(0, 1, 1000)\ns = np.sin(2 * np.pi * t)   # signal continu (simulé)`,
"Échantillonnage":`t = np.linspace(0, 1, 10, endpoint=False)   # 10 instants de mesure\nx = np.sin(2 * np.pi * t)`,
"Valeurs":`X = x.reshape(1, 10, 1)   # (lot, échantillons, canal)`,
"Convolution":`layers.Conv1D(filters=3, kernel_size=3)   # 3 filtres de taille 3, aléatoires au départ`,
"ReLU":`layers.ReLU()   # ou Conv(..., activation="relu")`,
"Pooling":`layers.MaxPooling1D(pool_size=2)`,
"Empilement":`layers.Conv1D(filters=3, kernel_size=3)   # filters=3 : 3 filtres, donc 3 cartes`,
"Flatten":`layers.Flatten()   # cartes 3D, vecteur 1D`,
"Couche dense":`layers.Dense(3)   # z = W·x + b : W et b sont créés puis appris automatiquement`,
"Softmax":`layers.Softmax()   # ou Dense(3, activation="softmax")`,
};
function cd(m,nm){var c=CK[nm];if(!c)return "";return c?'<details class="code"'+""+'><summary>Côté programmation'+(nm=="Entraînement"?" · Partie 1 : apprentissage":nm=="Prédiction"?" · Partie 2 : test":" : l’instruction de cette étape")+'</summary><pre>'+esc(c)+"</pre></details>":""}
var FULL={sig:`import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ["Carré", "Sinus", "Pic"]

# 1) Préparer les données (travail humain)
t = np.linspace(0, 1, 10, endpoint=False)
x = np.where((t > 0.3) & (t < 0.7), 9.0, 0.0) / 9.0   # un carré
X = x.reshape(1, 10, 1)
y = np.array([0])   # étiquette : 0 = Carré

# 2) Concevoir l'architecture (travail humain)
model = keras.Sequential([
    layers.Input(shape=(10, 1)),
    layers.Conv1D(3, 3, activation="relu"),        # convolution 1D + ReLU (3 filtres)
    layers.MaxPooling1D(2),                        # pooling
    layers.Flatten(),                              # flatten
    layers.Dense(3, activation="softmax"),         # couche dense + softmax
])

# 3) Entraîner : le CNN apprend filtres, W et b
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(X, y, epochs=100, verbose=0)

# 4) Prédire
p = model.predict(X)
print(CLASSES[int(np.argmax(p))], p)
`};
function rndK(m){S[m].Ks[S[m].f]=rk(m)[0];R(m)}
function pre(m,n){S[m].Ks[S[m].f]=cp(D[m].pk[n]);R(m)}
var SIGTRAIN=`# ===== PARTIE 1 : APPRENTISSAGE SUPERVISÉ =====
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ["Carré", "Sinus", "Pic"]             # 0, 1, 2
t = np.linspace(0, 1, 10, endpoint=False)        # 10 échantillons

# 1) Données étiquetées : plusieurs signaux par classe (travail humain)
def carre(a, b, h=9): return np.where((t > a) & (t < b), h, 0)
def sinus(phase, amp=4.5): return 4.5 + amp * np.sin(2 * np.pi * (t + phase))
def pic(c, h=9): return h * np.exp(-((t - c) / 0.1) ** 2)

signaux = [carre(.3, .7), carre(.25, .65, 8), carre(.35, .75), carre(.2, .6, 7),
           carre(.4, .8, 8), carre(.3, .6), carre(.25, .7, 8), carre(.35, .7),
           sinus(0), sinus(.05, 4), sinus(.95), sinus(.1, 3.5), sinus(.9, 4), sinus(.15), sinus(.85, 3.5), sinus(.97),
           pic(.5), pic(.4, 8), pic(.6), pic(.45, 7), pic(.55), pic(.35, 8), pic(.65), pic(.5, 7)]
y = np.array([0] * 8 + [1] * 8 + [2] * 8)        # étiquettes
X = (np.array(signaux) / 9.0).reshape(-1, 10, 1) # normalisation + forme (lot, 10, 1)

# 2) Architecture du CNN : on la définit UNE seule fois ici
model = keras.Sequential([
    layers.Input(shape=(10, 1)),
    layers.Conv1D(3, 3, activation="relu"),      # convolution 1D + ReLU (3 filtres)
    layers.MaxPooling1D(2),                      # pooling
    layers.Flatten(),                            # flatten
    layers.Dense(3, activation="softmax"),       # couche dense + softmax (3 classes)
])

# 3) Entraînement : le CNN corrige filtres, W et b pour réduire l'erreur
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=200, shuffle=True)

# 4) Sauvegarde : le modèle garde les filtres, W et b appris
model.save("cnn_signaux.keras")`,SIGPRED=`# ===== PARTIE 2 : TEST (rien n'est redéfini) =====
import numpy as np
from tensorflow import keras

CLASSES = ["Carré", "Sinus", "Pic"]

# 1) On recharge le modèle déjà entraîné : couches + filtres, W et b appris
model = keras.models.load_model("cnn_signaux.keras")

# 2) Nouveau signal entré par l'utilisateur (jamais vu à l'entraînement)
x = np.array([0, 0, 0, 0, 9, 9, 9, 9, 0, 0]) / 9.0   # MÊME normalisation
x = x.reshape(1, 10, 1)

# 3) Prédiction : le modèle enchaîne seul convolution, ReLU, pooling, flatten, dense, softmax
p = model.predict(x)[0]                       # 3 probabilités
print(CLASSES[int(np.argmax(p))], p)          # classe la plus probable`;
CK["Entraînement"]=SIGTRAIN;CK["Prédiction"]=SIGPRED;FULL.sig=SIGTRAIN+"\n\n"+SIGPRED;
S.sig={step:0,pre:"Carré",n:10,Ks:rk("sig"),f:0,h:[],relu:true,type:"max",c:[0,0],p:[0,0],W:null,tr:0,lab:0,b:[[0],[0],[0]],T:null,dup:0};sigI();sigSet();tvInit();cur="sig";R("sig");
