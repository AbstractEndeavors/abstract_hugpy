# curation_utils.py
from __future__ import annotations
import os, json, re, math, string, statistics
from pathlib import Path

PUNCT = set(string.punctuation)

def _safe_load_json(p): 
    try: return json.load(open(p, "r", encoding="utf-8"))
    except: return {}
def _safe_load_text(p):
    try: return open(p, "r", encoding="utf-8").read()
    except: return ""

def _parse_srt(text: str):
    out=[]
    if not text.strip(): return out
    blocks=re.split(r"\n\s*\n", text.strip())
    ts=lambda t:int(t.split(":")[0])*3600+int(t.split(":")[1])*60+float(t.split(":")[2].replace(",",".")) 
    for b in blocks:
        lines=[x.strip() for x in b.splitlines() if x.strip()]
        m=re.search(r"(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)", " ".join(lines))
        if not m: continue
        start,end=ts(m.group(1)),ts(m.group(2))
        content=" ".join([ln for ln in lines if "-->" not in ln and not ln.isdigit()])
        if end>start: out.append({"start":start,"end":end,"text":content})
    return out

def _tok(s:str): 
    s=s.lower()
    s=re.sub(r"[^\w\s'!?]"," ",s)
    return [t.strip("'") for t in s.split() if t.strip("'")]

def _tfidf(segments):
    docs=[_tok(seg.get("text","")) for seg in segments]
    N=len(docs) or 1
    df={}
    for d in docs:
        for w in set(d): df[w]=df.get(w,0)+1
    idf={w: math.log((N+1)/(df[w]+1))+1 for w in df}
    tfidf=[]
    for d in docs:
        tf={}
        for w in d: tf[w]=tf.get(w,0)+1
        L=len(d) or 1
        tfidf.append({w:(tf[w]/L)*idf.get(w,1.0) for w in tf})
    return tfidf,idf

def _score_segments(segments, keywords):
    tfidf,_= _tfidf(segments)
    kw=[k.lower() for k in keywords]
    emph={"!","?","wow","omg","amazing","hilarious","incredible"}
    scored=[]
    for i,seg in enumerate(segments):
        text=seg.get("text","") or ""
        toks=_tok(text)
        kw_score=sum(1 for t in toks for k in kw if k and k in t)
        sal=sum(sorted(tfidf[i].values(), reverse=True)[:5])
        emph_score=sum(1 for t in toks if t in emph)+text.count("!")*0.5+text.count("?")*0.25
        dur=max(0.01, seg["end"]-seg["start"])
        dur_bonus=0.6 if 6<=dur<=30 else (0.2 if 3<=dur<=45 else -0.2)
        score=1.2*kw_score + 2.0*sal + emph_score + dur_bonus
        scored.append((score,seg))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def _windows(segments, min_len=12.0, max_len=45.0, step=4.0):
    wins=[]
    n=len(segments)
    for i in range(n):
        start=segments[i]["start"]; end=start; j=i; parts=[]
        while j<n and end-start<max_len:
            end=segments[j]["end"]; parts.append(segments[j]["text"])
            if end-start>=min_len:
                wins.append({"start":start,"end":end,"text":" ".join(parts)})
                break
            j+=1
    if segments:
        t=segments[0]["start"]; T=segments[-1]["end"]
        while t+min_len<=T:
            wins.append({"start":t,"end":min(t+max_len,T),"text":""}); t+=step
    seen=set(); uniq=[]
    for w in wins:
        key=(round(w["start"],2), round(w["end"],2))
        if key in seen or w["end"]<=w["start"]: continue
        seen.add(key); uniq.append(w)
    return uniq

def _score_windows(windows, keywords, idf_hint=None):
    kw=[k.lower() for k in keywords]
    out=[]
    for w in windows:
        toks=_tok(w["text"] or "")
        coverage=len({k for k in kw if any(k in t for t in toks)})/(len(kw) or 1)
        dur=w["end"]-w["start"]
        sweet=math.exp(-((dur-24.0)**2)/(2*7.0**2))  # Gaussian around 24s
        novelty=0.0
        if idf_hint and toks:
            uniq=set(toks)
            novelty=sum(idf_hint.get(t,1.0) for t in uniq)/len(uniq)
        score=2.2*coverage + 1.8*sweet + 1.0*novelty
        out.append((score,w))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _calc_sharpness(img_path):
    try:
        import cv2
        img=cv2.imread(str(img_path))
        if img is None: return 0.0
        import numpy as np
        return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    except Exception:
        return 0.0

def _colorfulness(img_path):
    try:
        from PIL import Image; import numpy as np
        arr=np.array(Image.open(img_path).convert("RGB"))
        rg=(arr[:,:,0]-arr[:,:,1]).astype(float)
        yb=(0.5*(arr[:,:,0]+arr[:,:,1])-arr[:,:,2]).astype(float)
        return float(np.std(rg)+np.std(yb))
    except Exception:
        return 0.0

def _pick_best_thumbs(paths, k=3):
    scored=[]
    for p in paths:
        if not os.path.exists(p): continue
        s=_calc_sharpness(p); c=_colorfulness(p)
        scored.append((0.7*s+0.3*c, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _,p in scored[:k]]

def _classify_category(keywords, title="", description=""):
    text=" ".join([*(k.lower() for k in keywords), (title or "").lower(), (description or "").lower()])
    rules=[
        ("Comedy", ["comedy","funny","skit","humor"]),
        ("Music", ["music","song","album","concert"]),
        ("News & Politics", ["news","politic","report","debate"]),
        ("Education", ["education","tutorial","lesson","howto"]),
        ("Gaming", ["game","gaming","esport","playthrough"]),
        ("Sports", ["sports","match","tournament","league"]),
        ("Science & Technology", ["tech","review","unboxing","product","ai","software","hardware"]),
        ("Travel", ["travel","tour","journey","trip"]),
        ("Food", ["food","cook","recipe","kitchen","restaurant"]),
        ("Film & Animation", ["film","movie","animation","cinema","sketch"]),
    ]
    for label, needles in rules:
        if any(n in text for n in needles): return label
    return "Entertainment"

def _make_hashtags(keywords, clip_text, title=None, limit=12):
    toks=_tok((clip_text or "") + " " + (title or ""))
    stop={"the","and","to","of","a","in","is","it","that","for","on","with","as","this","at","be","are","was","were","an"}
    freq={}
    for t in toks:
        if t in stop or len(t)<3 or t.isdigit(): continue
        freq[t]=freq.get(t,0)+1
    top=[w for w,_ in sorted(freq.items(), key=lambda x:x[1], reverse=True)[:8]]
    base=[]
    for k in keywords:
        k=re.sub(r"[^A-Za-z0-9]+","",k)
        if len(k)>=3: base.append(k)
    base.extend(top)
    seen=set(); tags=[]
    for b in base:
        b=b[:40]
        if b.lower() in seen: continue
        tags.append("#"+b[0].upper()+b[1:])
        seen.add(b.lower())
        if len(tags)>=limit: break
    return tags

def aggregate_and_curate(base_dir: str|Path):
    base_dir=Path(base_dir)
    info      = _safe_load_json(base_dir/"video_info.json")
    meta      = _safe_load_json(base_dir/"video_metadata.json")
    whisper   = _safe_load_json(base_dir/"whisper_result.json")
    thumbs    = _safe_load_json(base_dir/"thumbnails.json")
    totals    = _safe_load_json(base_dir/"total_info.json")
    srt_txt   = _safe_load_text(base_dir/"captions.srt")

    # union keywords
    keywords=set()
    for src in (meta, info):
        for k in ("keywords","tags","categories"):
            v=src.get(k)
            if isinstance(v,list): keywords.update([str(x).strip().lower() for x in v])
            elif v: keywords.add(str(v).strip().lower())

    # segments
    segs=[]
    for s in whisper.get("segments",[]) or []:
        st, en = float(s.get("start",0)), float(s.get("end",0))
        if en>st: segs.append({"start":st,"end":en,"text":str(s.get("text","")).strip()})
    segs.extend(_parse_srt(srt_txt))
    segs.sort(key=lambda x:(x["start"],x["end"]))

    # clip
    seg_scored=_score_segments(segs, list(keywords))
    _,idf=_tfidf(segs)
    win_scored=_score_windows(_windows(segs,12.0,45.0,4.0), list(keywords), idf_hint=idf)
    best_clip=win_scored[0][1] if win_scored else (seg_scored[0][1] if seg_scored else {"start":0,"end":0,"text":""})
    candidates=[{"score":float(s), **w} for s,w in win_scored[:5]]

    # thumbs
    thumb_candidates=[]
    if meta.get("thumbnail_url"): thumb_candidates.append(meta["thumbnail_url"])
    if isinstance(thumbs.get("thumbnail_paths"), list): thumb_candidates+=thumbs["thumbnail_paths"]
    if (base_dir/"thumb.jpg").exists(): thumb_candidates.append(str(base_dir/"thumb.jpg"))
    ranked_thumbs=_pick_best_thumbs(thumb_candidates, 3)

    # title/desc
    title=meta.get("title") or info.get("title")
    description = meta.get("description") or meta.get("summary") or (best_clip.get("text","")[:300]+"..." if best_clip.get("text") else None)

    # category + hashtags
    category = meta.get("category") or _classify_category(list(keywords), title or "", description or "")
    hashtags = _make_hashtags(sorted(list(keywords)), best_clip.get("text",""), title)

    result={
        "id": info.get("id"),
        "title": title,
        "description": description,
        "keywords": sorted(list(keywords)),
        "category": category,
        "canonical_url": meta.get("seodata",{}).get("seo_data",{}).get("canonical_url") or info.get("webpage_url"),
        "duration": meta.get("seodata",{}).get("seo_data",{}).get("duration_formatted") or info.get("duration"),
        "uploader": meta.get("seodata",{}).get("seo_data",{}).get("uploader",{}),
        "publication_date": meta.get("seodata",{}).get("seo_data",{}).get("publication_date"),
        "video_path": totals.get("video_path") or info.get("file_path"),
        "audio_path": totals.get("audio_path"),
        "transcript_excerpt": (best_clip.get("text","")[:500] + ("..." if len(best_clip.get("text",""))>500 else "")) if best_clip else "",
        "thumbnails_ranked": ranked_thumbs,
        "best_clip": best_clip,
        "candidate_clips": candidates,
        "hashtags": hashtags,
        "schema_markup": meta.get("seodata",{}).get("seo_data",{}).get("schema_markup"),
        "social_metadata": meta.get("seodata",{}).get("seo_data",{}).get("social_metadata"),
        "source_flags": totals,
    }
    return result



### Bash runner (you said you prefer bash)

