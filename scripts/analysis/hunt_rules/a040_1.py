import pandas as pd
def bets(df, odds=None):
    """準優勝戦・優勝戦で、1号艇の展示タイムが最速(er1==1・締切前情報)かつ1コース1着率70%以上なら単勝1を100円"""
    n = df.race_name.fillna('').str.replace(r'[\s　]', '', regex=True)
    semi = n.str.contains('準優勝戦') & ~n.str.contains('準々')
    final = n.str.contains('優勝戦') & ~n.str.contains('準')
    m = (semi | final) & (df.er1 == 1) & (df.cw1 >= 70)
    r = df.loc[m, ['hd', 'jcd', 'rno']].copy()
    r['ticket_type'] = 'tansho'; r['combination'] = '1'; r['amount'] = 100
    return r
