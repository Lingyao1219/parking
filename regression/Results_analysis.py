import pandas as pd
from functools import reduce


def format_table2(filename=' Restaurant _Gam_Pop_Pct_scale.csv', Yname='Case-fatality ratio'):
    All_corr1 = pd.read_csv(r'D:\Google_Review\Parking\results\\' + filename)
    All_corr1['conf.low'] = All_corr1['Estimate'] - 1.96 * All_corr1['Std..Error']
    All_corr1['conf.high'] = All_corr1['Estimate'] + 1.96 * All_corr1['Std..Error']
    # All_corr1 = All_corr1.replace({"names": di_replace_nn})
    All_corr1['Symbol_'] = ' '
    All_corr1.loc[All_corr1['Pr...t..'] <= 0.001, 'Symbol_'] = '***'
    All_corr1.loc[(All_corr1['Pr...t..'] <= 0.01) & (All_corr1['Pr...t..'] > 0.001), 'Symbol_'] = '**'
    All_corr1.loc[(All_corr1['Pr...t..'] <= 0.05) & (All_corr1['Pr...t..'] > 0.01), 'Symbol_'] = '*'
    # All_corr1.loc[(All_corr1['Pr...t..'] <= 0.1) & (All_corr1['Pr...t..'] > 0.05), 'Symbol_'] = '.'
    for jj in list(All_corr1.columns):
        try:
            All_corr1[jj] = All_corr1[jj].round(3).map('{:.3f}'.format).astype(str)  # .apply('="{}"'.format)
        except:
            All_corr1[jj] = All_corr1[jj]
    n_cc = ['Adj_R2', 'ti(Lat,Lng)', 's(CBSA)', 'dev.expl', 'n']
    All_corr1.loc[~All_corr1['names'].isin(n_cc), 'Estimate'] = \
        All_corr1.loc[~All_corr1['names'].isin(n_cc), 'Estimate'] + \
        All_corr1.loc[~All_corr1['names'].isin(n_cc), 'Symbol_'] + '\n(' + \
        All_corr1.loc[~All_corr1['names'].isin(n_cc), 'conf.low'] + ', ' + \
        All_corr1.loc[~All_corr1['names'].isin(n_cc), 'conf.high'] + ')'

    # Return
    All_corr_final = All_corr1[['names', 'Estimate']]
    # All_corr_final.to_excel(r'D:\\Vaccination\\Results\\' + outname)
    All_corr_final.columns = ['Variables', Yname]
    All_corr_final = All_corr_final[All_corr_final['Variables'] != 'Blank']
    return All_corr_final


gam_all = format_table2(filename=' all _Gam_Pop_Pct_scale.csv', Yname='All')
gam_res = format_table2(filename=' Restaurant _Gam_Pop_Pct_scale.csv', Yname='Restaurant')
gam_rt = format_table2(filename=' Retail Trade _Gam_Pop_Pct_scale.csv', Yname='Retail Trade')
gam_re = format_table2(filename=' Recreation _Gam_Pop_Pct_scale.csv', Yname='Recreation')
gam_hotel = format_table2(filename=' Hotel _Gam_Pop_Pct_scale.csv', Yname='Hotel')
gam_ps = format_table2(filename=' Personal Service _Gam_Pop_Pct_scale.csv', Yname='Personal Service')
gam_apt = format_table2(filename=' Apartment _Gam_Pop_Pct_scale.csv', Yname='Apartment')
cross_coeff_all = reduce(lambda left, right: pd.merge(left, right, on=['Variables'], how='outer'),
                         [gam_all, gam_res, gam_rt, gam_re, gam_hotel, gam_ps, gam_apt])
cross_coeff_all.to_excel(r'D:\Google_Review\Parking\results\\Gam_all_cross_coeff.xlsx')
