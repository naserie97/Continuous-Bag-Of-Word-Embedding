install: 
	pip install --upgrade pip &&\
		pip install -r requirments.txt


format:
	black *.py 