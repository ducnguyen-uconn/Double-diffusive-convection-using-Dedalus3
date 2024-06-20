.PHONY: clean
clean: 
	rm -rf $(OUTPUT)

# Salt-Finger Convection
exportPDF_SFC:
	jupyter nbconvert --to pdf Salt-finger-conection_sim_example.ipynb
run_SFC:
	jupyter nbconvert --to script Salt-finger-conection_sim_example.ipynb
	python3 Salt-finger-conection_sim_example.py

# Doubly-periodic Salt-Finger Convection using single-mode equations
pdfsm:
	jupyter nbconvert --to pdf single-mode.ipynb
runsm:
	jupyter nbconvert --to python single-mode.ipynb
	python3 single-mode.py

