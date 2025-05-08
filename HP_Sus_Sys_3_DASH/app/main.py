import sys, os 
from sentence_transformers import SentenceTransformer
from manifest_generation import *
from LLM import *
import subprocess
from testing import *
from CSV_upload import *
import time
import requests

from detect_os import detect_os
user_os = detect_os()

questions = ["\n \n \n Can you tell me how much carbon emission is produced by machine ld71r18u44dws?\n", 
                "How much is the total carbon emissions for all the machines? (dynamic programming) \n", 
                "Which machine has the GPU highest average utilisation?\n", 
                "Give me a summary of the central processing unit usage for all the machines\n",
                "Which of the machines do you recommend being moved to the next level up of compute power and why?\n",
                "What is the central processing unit average utilisation for each machine?\n",
                "What machine has the highest carbon emission value?\n",
                "How much is the total carbon emissions for all the machines?\n"]

def start_server():
    """
    Starts Ollama server, checking the OS
    """
    logs = []

    if user_os == "Windows":
        script_name = 'start_server.bat'
    else:  # macOS or Linux
        script_name = 'start_server.sh'
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        logs.append(f"\n\n⚠️  Script to start Ollama server not found: {script_path}")
        return "\n".join(logs)  
        
    logs.append("⏳ Starting Ollama server...")
    # print("Please wait for the server to start... \n\n")
    if user_os == "Windows":
        subprocess.Popen([script_path], shell=True)
    else:
        subprocess.Popen(['sh', script_path])
    # Wait for the server to start
    time.sleep(5)  # Adjust this delay if needed
    return "\n".join(logs)
    

def check_server():
    try:
        response = requests.get('http://127.0.0.1:11434')  # Adjust URL and port as needed
    except requests.RequestException as e:
        raise ConnectionError(f"Could not connect to Ollama: {e}")
    
    if response.status_code == 200:
        return "\n\n★ ☆ ★ ☆ Server is up and running ★ ☆ ★ ☆ \n\n"
    else:
        raise RuntimeError(f"Server returned status code {response.status_code}.\n\n")

def prepare_chatbot():
    # If the user does not input any file path, the default test file path will be used
    default_file_path = r"data/1038-0610-0614-day-larger-figures-test.xlsx"
    target_dir = r"data/uploaded_excel_files"
    uploaded_file_path = upload_file_to_application_directory(target_dir, default_file_path=default_file_path)

    # Check that the user hasn't quit, and/or that the file was uploaded correctly!
    if uploaded_file_path is None:
        raise RuntimeError("No file uploaded")
    # Initial pipeline for Impact Framework
    # Define the input file path - need to work out how this will work if it's uploaded by the user
    excel_file = uploaded_file_path
    print(excel_file)

    # Convert the Excel file to a CSV file
    csv_file = convert_xlsx_to_csv(excel_file)

    # Define the input and output file paths
    original_CSV_filepath = csv_file
    modified_CSV_filepath = r'data/modified_CSV.csv'
    manifest_filepath = r'manifest1/z2_G4_Sci.yaml'

    # Process the CSV file and extract the duration value, start date, end date, and templates to create the manifest file
    try:
    # Your main code execution
        modified_csv_path, duration, start_date, end_date, templates, analysis_window = process_csv(original_CSV_filepath, modified_CSV_filepath)
    except ValueError as e:
        raise ValueError(f"Incorrect file structure: {e}")

    try:
        manifest_success = safe_generate_manifest(manifest_filepath, modified_csv_path, duration, templates)

        yield "\n\n★ ☆ ★ ☆ Attempting to generate manifest file with telemetry data... ★ ☆ ★ ☆\n\n"

        safe_print_file_info(modified_CSV_filepath, "Modified CSV file")
        safe_print_file_info(manifest_filepath, "Manifest file")

        if duration:
            print(f"\nThis telemetry data was observed over a period of: {duration} seconds")
        else:
            print("\nWarning: Duration value was not extracted successfully")

        current_dir = os.getcwd()
        print(f"\nCurrent working directory: {current_dir}")

        # Construct absolute paths
        manifest_path = os.path.abspath(os.path.join(current_dir, 'manifest1', 'z2_G4_Sci.yaml'))
        output_path = os.path.abspath(os.path.join(current_dir, 'manifest1', 'outputs', 'z2_G4_Sci_Output.yaml'))
  
        # Check if paths exist
        if os.path.exists(manifest_path):
            print(f"\nManifest file found at: {manifest_path}")
        else:
            print(f"\nWarning: Manifest file not found at: {manifest_path}")

        if os.path.exists(output_path):
            print(f"\nOutput directory found at: {output_path}")

        # If manifest generation was successful, try to read and print some info
        if manifest_success:
            try:
                with open(manifest_filepath, 'r') as f:
                    manifest_data = yaml.safe_load(f)
                print(f"\nManifest file successfully read. Contains {len(manifest_data)} top-level keys.")
            except Exception as e:
                print(f"Warning: Could not read manifest file: {str(e)}")

    except Exception as e:
        raise RuntimeError("Unexpected error")
        

    # Construct the command with absolute paths
    command = f'if-run --manifest "{manifest_path}" --output "{output_path}"'
    # Run the terminal command
    # command = r"ie --manifest '\manifest1\NEW_z2_G4_Sci.yaml' --output '\manifest1\outputs\z2_G4_Sci_Output'"

    yield "\n\n★ ☆ ★ ☆ Running Impact Framework command... ★ ☆ ★ ☆\n\n"
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        yield "Command output:"
        yield result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Impact Framework failed: {e.stderr}")

    # Pipeline to run after running the termimnal commmand to run the Impact Framework
    # taking in our raw 'uploaded.xlsx' file
    excel_file = excel_file
    # taking in the output yaml file with the carbon emissions data from IF
    yaml_file = r'manifest1/outputs/z2_G4_Sci_Output.yaml'

    try:
        # Load and prepare the Excel file
        prepared_df = prepare_excel_file(excel_file)

        # Load emissions reference data and perform further operations
        emissions_reference_data = load_data_files(yaml_file)
        machine_emissions_list, machine_id_dict = extract_data_from_yaml(emissions_reference_data)

        # Merge data into a single DataFrame
        merged_df, machine_ids = merge_data_into_one_df(prepared_df, machine_emissions_list, machine_id_dict)

    except ValueError as e:
        print(f"Error: {e}. Please upload a correctly formatted file.")

    
    columns_to_exclude = ['model', 'timestamp', 'Machine', 'number of cores']
    columns_to_label = [col for col in merged_df.columns if col not in columns_to_exclude]
   
    convertible_columns = []

    # Try to convert each column to float but skip if forr some reason column is not numeric
    for col in columns_to_label:
        try:
            merged_df[col] = merged_df[col].astype(float)
            convertible_columns.append(col)
        except ValueError:
            print(f"Column {col} could not be converted to float and will be skipped.")

    for col in convertible_columns:
        merged_df[col] = label_max_min(merged_df[col])

    # Append the total carbon emissions row
    # merged_df = append_sum_row(merged_df, 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS')
    # print(merged_df.columns)

    # Save the merged DataFrame to a CSV file 
    merged_df.to_csv(r'embeddings/merged_df.csv', index=False)

    csv_filename = r'embeddings/merged_df.csv'

    # convert the csv file to a json file
    data_dict_json = csv_to_json(csv_filename, as_json=False)

    # Flatten the dictionary and stringify it for our sentences    
    flat_dict = flatten(data_dict_json)
    dict_string = stringify(flat_dict)

    with open(r'embeddings/data.txt', 'w') as f:
        f.write(dict_string)

    # Read the stringified flat dictionary from the file
    with open(r'embeddings/data.txt', 'r') as f:
        read_back_string = f.read()

    print("\nStringified flat dictionary read back from the file.")
    # print(read_back_string)

    # Path to data.txt file
    sentences_file_path = r'embeddings/data.txt'

    # Read sentences from file
    sentences = read_sentences_from_file(sentences_file_path)
    add_context_to_sentences(sentences, duration, start_date, end_date, analysis_window, num_of_machines=str(len(machine_ids)), merged_df=merged_df)
    # print(sentences)

    # Load the pre-trained model for embedding with SentenceTransformer
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

    # Embed the sentences using the model
    index, embeddings = embed_sentences(sentences, model)

  

    # Generate question function contains the dynamic programming approach for calculations
    # generate_question(index, embeddings, model, sentences, questions, machine_ids, model_name)
    # The process user input function contains a wrapper function instead which more flexibly performs calculations for more data points, the llm can judge its applicability for itself. can be applied to any machines not just a 'total' value for all machines
    # process_user_input(machine_ids, model, index, sentences, send_prompt, questions)

    return machine_ids,model,index,sentences,send_prompt,questions

    

