-- 初期スキーマ。項目が確定したら raw カラムを分解して 0002 以降で追加する。

create table if not exists racelist (
  hd    text not null,          -- 開催日 YYYYMMDD
  jcd   text not null,          -- 場コード 01-24
  rno   int  not null,          -- レース番号
  lane  int  not null,          -- 艇番 1-6
  raw   jsonb,                  -- 出走表の生テキスト（項目確定まで）
  fetched_at timestamptz default now(),
  primary key (hd, jcd, rno, lane)
);

create table if not exists results (
  hd    text not null,
  jcd   text not null,
  rno   int  not null,
  raw_order  jsonb,             -- 着順表の生テキスト
  raw_payout jsonb,             -- 払戻表の生テキスト
  fetched_at timestamptz default now(),
  primary key (hd, jcd, rno)
);

create table if not exists picks (
  hd    text not null,
  jcd   text not null,
  rno   int  not null,
  rules_version int,
  bet_type text,
  pattern  text,
  unit_yen int,
  reason   text,
  created_at timestamptz default now(),
  primary key (hd, jcd, rno)
);

create table if not exists verifications (
  hd    text primary key,
  races int,
  hits  int,
  hit_rate     numeric,
  bet_total    int,
  payout_total int,
  roi          numeric,
  created_at timestamptz default now()
);
