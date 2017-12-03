init:
	pip install -r requirements

localinit:
	pip install --editable .

test:
	cd tests && nosetests

.PHONY: init test 
