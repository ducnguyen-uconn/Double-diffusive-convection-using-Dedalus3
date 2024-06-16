.PHONY: clean
clean: 
	rm -rf $(OUTPUT)

# Salt-Finger Convection
PDF_DNS:
	jupyter nbconvert --to pdf SFC_sim_example.ipynb
run_DNS:
	jupyter nbconvert --to script Salt-finger-conection_sim_example.ipynb
	python3 Salt-finger-conection_sim_example.py


PDF_SME:
	jupyter nbconvert --to pdf Salt-finger_convection_using_single-mode_equations.ipynb
run_SME:
	jupyter nbconvert --to script Salt-finger_convection_using_single-mode_equations.ipynb
	python3 Salt-finger_convection_using_single-mode_equations.py


