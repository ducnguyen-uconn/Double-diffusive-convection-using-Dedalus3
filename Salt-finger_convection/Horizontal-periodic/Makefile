.PHONY: clean
clean: 
	rm -rf $(OUTPUT)

# Salt-Finger Convection
pdfdns:
	jupyter nbconvert --to pdf dns.ipynb
rundns:
	jupyter nbconvert --to python dns.ipynb
	python3 dns.py


pdfsm:
	jupyter nbconvert --to pdf single-mode.ipynb
runsm:
	jupyter nbconvert --to python single-mode.ipynb
	python3 single-mode.py


