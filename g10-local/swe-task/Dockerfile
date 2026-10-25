FROM python:3.12-slim
WORKDIR /work
COPY calc.py test_calc.py ./
CMD ["python", "-m", "unittest", "test_calc.py"]
