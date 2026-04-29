# Use base image AWS for python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy requirements.txt file and enable root task lambda to indicate pre-require task
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the libs
RUN pip install -r requirements.txt

# Copy PDF file
COPY Clean-Architecture.pdf ${LAMBDA_TASK_ROOT}/Clean-Architecture.pdf

# Copy the function code
COPY simplerag.py ${LAMBDA_TASK_ROOT}

# Set permission to make the file executable
RUN chmod +x simplerag.py

# Set CMD to handler
CMD ["simplerag.lambda_handler"]