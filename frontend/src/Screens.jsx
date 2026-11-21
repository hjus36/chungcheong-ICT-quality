import React from 'react';

const examples = [
  ['SYN-0042', '08:41', 0.91, 2], ['SYN-0017', '08:16', 0.64, 1],
  ['SYN-0088', '09:27', 0.31, 1], ['SYN-0003', '08:02', 0.18, 0],
  ['SYN-0061', '09:00', 0.09, 0], ['SYN-0110', '09:49', 0.04, 0]
];

export function Home({go, example, busy}) {
  return <div className="landing">
    <section className="hero"><h1>사출성형 품질위험 검토</h1><p>사이클 데이터를 올리고, 먼저 검사할 부품을 확인하세요.</p>
      <div className="hero-actions"><button className="primary" disabled={busy} onClick={()=>go('upload')}>데이터 업로드</button><button disabled={busy} onClick={example}>합성 예제 결과 보기 ›</button></div></section>
    <section className="showcase"><div><h2>부품 검토</h2><p>위험 점수가 높은 부품부터<br/>참고 구간을 벗어난 공정변수까지.</p><button disabled={busy} onClick={example}>합성 예제 대시보드 ›</button></div>
      <div className="preview-sheet"><h3>검토 목록 <small>화면 예시 · 6건</small></h3><div className="table-scroll"><table><thead><tr>{['부품 ID','사이클 시작','위험 점수','검토 구분','참고 구간 이탈'].map(x=><th key={x}>{x}</th>)}</tr></thead>
      <tbody>{examples.map(([id,time,v,n])=><tr key={id}><td>{id}</td><td>{time}</td><td>{v.toFixed(2)}</td><td>{v>=0.25?'우선 검토':'통상 검토'}</td><td>{n}개</td></tr>)}</tbody></table></div></div></section>
    <div className="landing-tiles"><section><h2>공정변수 분석</h2><p>보압·스트로크 등 공정변수의 분포를 살펴보세요.</p><button disabled={busy} onClick={()=>go('variables')}>변수 분포 보기 ›</button><div className="hist-art" aria-hidden="true">{[12,23,40,61,88,110,139,159,151,131,106,80,58,35,22,12].map((h,i)=><i key={i} style={{height:h}}/>)}</div><small>공정변수 분포 예시</small></section>
      <section><h2>모델 검증</h2><p>모델 선택 기록과 기준선 비교를 확인하세요.</p><button disabled={busy} onClick={()=>go('model')}>검증 결과 보기 ›</button><div className="matrix-art">{['통상 검토','오경고','위험 누락','위험 탐지'].map(k=><b key={k}>{k}</b>)}</div><small>혼동행렬 구성 예시</small></section></div>
  </div>
}

export function ReviewPanel({row, fmt, date, label, onDetail}) {
  if(!row)return null;
  const deviations=row.outside_training_reference||[];
  return <aside className="review-side"><div className="review-top"><small>선택한 부품</small><h2>{row.part_id??row.batch_id}</h2><p>사이클　{date(row.cycle_start_at)} – {date(row.cycle_end_at)}{row.material?`　·　${row.material}`:''}</p>
    <div className="readout"><div>모델 위험 점수<strong>{fmt(row.risk_score)}</strong><b>{row.review_priority}</b></div><div className="gauge" aria-hidden="true" style={{'--score':(Math.max(0,Math.min(1,row.risk_score))*100)+'%'}}><span>0 – 1</span></div></div><small>검토 우선순위 참고용 · 보정된 추정 확률</small></div>
    <div className="review-bottom"><h3>참고 구간을 벗어난 항목</h3><small>학습 정상 부품 5~95% 구간 · 공정 규격 아님</small>{deviations.slice(0,3).map(d=><p key={d.feature}>{label(d.feature)}<b>{d.direction||'이탈'}</b></p>)}{!deviations.length&&<p>참고 구간 이탈 없음</p>}<button onClick={()=>onDetail(row)}>부품 상세 보기 ›</button></div></aside>
}
