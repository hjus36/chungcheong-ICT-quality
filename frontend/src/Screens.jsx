import React from 'react';

const examples = [
  ['HT-1112-04', '10:20', 0.86, 2], ['HT-1112-02', '09:10', 0.72, 1],
  ['HT-1112-05', '11:00', 0.32, 1], ['HT-1112-01', '08:30', 0.29, 0],
  ['HT-1112-03', '09:45', 0.21, 0], ['HT-1112-06', '11:40', 0.15, 0]
];

export function Home({go, example, busy}) {
  return <div className="landing">
    <section className="hero"><h1>열처리 공정 품질 분석</h1><p>작업 데이터를 올리고, 먼저 검토할 공정을 확인하세요.</p>
      <div className="hero-actions"><button className="primary" disabled={busy} onClick={()=>go('upload')}>데이터 업로드</button><button disabled={busy} onClick={example}>합성 예제 결과 보기 ›</button></div>
      <img src="/heat-rings.svg" width="900" height="325" alt="열처리 금속 링 세 개"/></section>
    <section className="showcase"><div><h2>작업 검토</h2><p>위험 점수가 높은 작업부터<br/>참고 구간을 벗어난 항목까지.</p><button disabled={busy} onClick={example}>합성 예제 대시보드 ›</button></div>
      <div className="preview-sheet"><h3>검토 목록 <small>화면 예시 · 6건</small></h3><div className="table-scroll"><table><thead><tr>{['작업번호','공정 시작','위험 점수','검토 구분','참고 구간 이탈'].map(x=><th key={x}>{x}</th>)}</tr></thead>
      <tbody>{examples.map(([id,time,v,n])=><tr key={id}><td>{id}</td><td>{time}</td><td>{v.toFixed(2)}</td><td>{v>=0.35?'우선 검토':'통상 검토'}</td><td>{n}개</td></tr>)}</tbody></table></div></div></section>
    <div className="landing-tiles"><section><h2>공정변수 분석</h2><p>온도와 센서값의 분포를 살펴보세요.</p><button disabled={busy} onClick={()=>go('variables')}>변수 분포 보기 ›</button><div className="hist-art" aria-hidden="true">{[12,23,40,61,88,110,139,159,151,131,106,80,58,35,22,12].map((h,i)=><i key={i} style={{height:h}}/>)}</div><small>센서값 분포 예시</small></section>
      <section><h2>모델 검증</h2><p>예측과 실제 라벨을 비교해 성능을 확인하세요.</p><button disabled={busy} onClick={()=>go('model')}>검증 결과 보기 ›</button><div className="matrix-art">{['통상 검토','오경고','위험 누락','위험 탐지'].map(k=><b key={k}>{k}</b>)}</div><small>혼동행렬 구성 예시</small></section></div>
  </div>
}

export function ReviewPanel({row, fmt, date, label, onDetail}) {
  if(!row)return null;
  const deviations=row.outside_training_reference||[];
  return <aside className="review-side"><div className="review-top"><div className="heat-lines" aria-hidden="true">{Array.from({length:10},(_,i)=><img key={i} src="/heat-curve.svg" alt="" style={{right:18-i*6,top:-33+i*5}}/>)}</div><small>선택한 작업</small><h2>{row.batch_id}</h2><p>공정 시간　{date(row.process_start_at)} – {date(row.process_end_at)}</p>
    <div className="readout"><div>모델 위험 점수<strong>{fmt(row.risk_score)}</strong><b>{row.review_priority}</b></div><div className="gauge" aria-hidden="true" style={{'--score':(Math.max(0,Math.min(1,row.risk_score))*100)+'%'}}><span>0 – 1</span></div></div><small>검사 우선순위 참고용 · 불량 확률 아님</small></div>
    <div className="review-bottom"><h3>참고 구간을 벗어난 항목</h3><small>학습 5~95% 구간 기준 · 기업 규격 아님</small>{deviations.slice(0,3).map(d=><p key={d.feature}>{label(d.feature)}<b>{d.direction||'이탈'}</b></p>)}{!deviations.length&&<p>참고 구간 이탈 없음</p>}<button onClick={()=>onDetail(row)}>전체 작업 상세 보기 ›</button></div></aside>
}
