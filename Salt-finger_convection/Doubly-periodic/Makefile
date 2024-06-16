.PHONY: clean
clean: 
	rm -rf $(OUTPUT)

# Reyleigh Benard Convection
exportPDF_RBC:
	jupyter nbconvert --to pdf RBC_sim_example.ipynb
run_RBC:
	jupyter nbconvert --to script RBC_sim_example.ipynb
	python3 RBC_sim_example.py

# Salt-Finger Convection
exportPDF_SFC:
	jupyter nbconvert --to pdf Salt-finger-conection_sim_example.ipynb
run_SFC:
	jupyter nbconvert --to script Salt-finger-conection_sim_example.ipynb
	python3 Salt-finger-conection_sim_example.py



