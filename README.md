# Exercise-module9
Exercise modul 9 tentang model selection dan deployment

# Input
Contoh input format:

	- Input berupa dictionary yang merupakan  
	sample data: 
  
	{
    "person_age": 30,
    "person_income": 66120,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 10.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "loan_amnt": 28000,
    "loan_int_rate": 13.57,
    "loan_percent_income": 0.04,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6
	}

# Output
Contoh output format:

	- Dictionary
	{
    "api_version": "v1",
    "model": "LR",
    "result": "0.285"
	}

# http method 
yang digunakan ialah POST dengan mengirimkan data tanpa melalui url
	
	
