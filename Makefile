## Sample makefile from http://docs.python-guide.org/en/latest/writing/structure/

init:
	pip install -r requirements.txt

test:
	nosetools tests
