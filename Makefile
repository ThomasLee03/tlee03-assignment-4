install:
ifeq ($(OS),Windows_NT)
	python -m venv venv
	venv\Scripts\activate && pip install -r requirements.txt
else
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
endif

run:
ifeq ($(OS),Windows_NT)
	venv\Scripts\activate && flask run --host=0.0.0.0 --port=3000
else
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000
endif