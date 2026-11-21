import pandas as pd
def bets(df, odds=None):
    """準優勝戦・優勝戦で、1号艇が40歳以下・1コース1着率70%以上・モーター2連率35%以上なら単勝1を100円(前夜情報のみ)"""
    n = df.race_name.fillna('').str.replace(r'[\s　]', '', regex=True)
    semi = n.str.contains('準優勝戦') & ~n.str.contains('準々')
    final = n.str.contains('優勝戦') & ~n.str.contains('準')
    m = (semi | final) & (df.age1 <= 40) & (df.cw1 >= 70) & (df.mo1 >= 35)
    r = df.loc[m, ['hd', 'jcd', 'rno']].copy()
    r['ticket_type'] = 'tansho'; r['combination'] = '1'; r['amount'] = 100
    return r
