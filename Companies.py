# Вспомогательный файл, содержащий список компаний и их альтернативные названия

def get_forbes_comp():
    consumer_fin_services = ['American Express', 'Capital One', 'Visa', 'Mastercard',
                             'Discover Financial Services', 'CIT', 'Sallie Mae']
    diversified_ins = ['MetLife', 'American International Group', 'Allstate', 'The Hartford', 'Loews',
                       'Reinsurance Group of America', 'Genworth', 'Assurant', 'Markel',
                       'American Financial Group', 'Cincinnati Financial']
    ins_brokers = ['Marsh & McLennan', 'Gallagher']
    inv_services = ['Berkshire', 'FNMA', 'Freddie Mac', 'Goldman Sachs', 'Morgan Stanley', 'BlackRock',
                    'Ameriprise Financial', 'State Street', 'Charles Schwab', 'Voya Financial',
                    'Raymond James', 'Intercontinental Exchange', 'Franklin Resources',
                    'TD Ameritrade', 'T. Rowe Price', 'Invesco', 'CME',
                    'E-Trade', 'Interactive Brokers', 'New Residential Inv',
                    'AGNC', 'Ares Capital']
    life_health_ins = ['Prudential Financial', 'Aflac', 'Lincoln National', 'Principal Financial',
                       'Unum', 'Brighthouse Financial', 'Torchmark', 'American Equity Investment']
    major_banks = ['JPMorgan', 'Bank of America', 'Wells Fargo', 'Citigroup',
                   'PNC', 'Bank of New York Mellon', 'BB&T', 'KeyCorp', 'Regions Financial',
                   'Comerica']
    property_casualty_ins = ['Progressive', 'Travelers', 'WR Berkley', 'Fidelity National Financial',
                             'Alleghany', 'Old Republic International']
    thrifts_mort_fin = ["People's United", 'New York Community Bancorp', 'IBERIABANK']

    return consumer_fin_services + diversified_ins + ins_brokers + inv_services + life_health_ins + \
           major_banks + property_casualty_ins + thrifts_mort_fin


def alternative_names():
    return {'JPMorgan': ['JP Morgan', 'J.P. Morgan', 'J.P.Morgan'],
            'Citigroup': ['Citibank'],
            'Sallie Mae': ['SLM Corporation'],
            'Bank of America': ['BofA', 'BAC'],
            'American International Group': ['AIG'],
            'US Bancorp': ['U.S. Bancorp', 'U.S.Bancorp'],
            'WR Berkley': ['W.R. Berkley', 'W.R.Berkley'],
            'Bank of New York Mellon': ['BNY Mellon'],
            'Berkshire': ['Buffett'],
            'FNMA': ['Fannie Mae', 'Federal National Mortgage Association']}

