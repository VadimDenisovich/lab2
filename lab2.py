import scipy.stats as stats
import pandas as pd 
import numpy as np 

SEED = 3
SIZE_100 = 100
SIZE_1000 = 1000 
SIZES = ['100', '1000']

# Нормальное распределение с параметрами mu=1, sigma=4, 
# где mu — матожидание, а sigma^2 — дисперсия. 
# Поскольку sigma здесь это также стандартное отклонение, то большая часть значений
# будет находится в пределах +-4стандартных отклоения от среднего. 
norm_100 = stats.norm.rvs(loc=1, scale=4, size=SIZE_100, random_state=SEED)
norm_1000 = stats.norm.rvs(loc=1, scale=4, size=SIZE_1000, random_state=SEED)

# Истинные параметры 
true_mu = 1 
true_sigma = 4 

def calculate_norm_estimates(sample): 
	# Метод Моментов (ММ), оценки: 
	mm_mu = np.mean(sample)
	mm_sigma = np.std(sample, ddof=0) # ddof=0 потому что оценка смещённая 

	# Метод максимального правдоподобия (ММП) — Maximum likelihood method, оценки: 
	mle_mu = np.mean(sample)
	# ddof=1 потому что оценка смещённая → учесть, что 
	# fit-оценка будет смещенной, там ddof=0. в
	mle_sigma = np.std(sample, ddof=0) 
	

	return mm_mu, mm_sigma, mle_mu, mle_sigma

# Список для оценок выборок разных размеров 
norm_results = []

for sample, size in zip([norm_100, norm_1000], SIZES): 
	mm_mu, mm_sigma, mle_mu, mle_sigma = calculate_norm_estimates(sample)

	# Fit методы для сравнения моих оценок.
	# Возвращает оценки mu и sigma методом моментов. 
	fit_mm_mu, fit_mm_sigma = stats.norm.fit(sample, method='MM')
	# Возвращает оценки mu и sigma методом максимального правдоподобия. 
	fit_mle_mu, fit_mle_sigma = stats.norm.fit(sample)
	
	norm_results.append({
		'Размер выборки': size, 
		'Истинное μ': true_mu,
		'Истинное σ': true_sigma,
		'ММ-оценка μ': mm_mu,
		'fit-MM-оценка μ': fit_mm_mu,
		'ММ-оценка σ': mm_sigma,
		'fit-MM-оценка σ': fit_mm_sigma,
		'ММП-оценка μ': mle_mu,
		'fit-ММП-оценка μ': fit_mle_mu,
		'ММП-оценка σ': mle_sigma, 
		'fit-ММП-оценка σ': fit_mle_sigma,
})

result_norm_df = pd.DataFrame(norm_results)
result_norm_df.to_csv('norm_distributions.csv', index=False)

# Распределение Бернулли с вероятностью 0.5 
bernoulli_100 = stats.bernoulli.rvs(p=0.5, size=SIZE_100, random_state=SEED)
bernoulli_1000 = stats.bernoulli.rvs(p=0.5, size=SIZE_1000, random_state=SEED)

# Истинное значение p 
true_p = 0.5 

def calculate_bernoulli_estimates(sample): 
	# ММ-оценка 
	mm_p = np.mean(sample)

	# ММП-оценка
	mle_p = mm_p # также равна среднему 

	return mm_p, mle_p

# Список для оценок 
bernoulli_results = [] 

for sample, size in zip([bernoulli_100, bernoulli_1000], SIZES):
    mm_p, mle_p = calculate_bernoulli_estimates(sample)

		# нет stats.bernoulli.fit 

    bernoulli_results.append({
        'Размер выборки': size,
        'Истинное p': true_p,
        'ММ-оценка p': mm_p,
        'ММП-оценка p': mle_p
    })

result_bernoulli_df = pd.DataFrame(bernoulli_results)
result_bernoulli_df.to_csv('bernoulli_distributions.csv', index=False)