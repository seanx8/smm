import myFeature
import myLanguageDetection
import myfile
import myplot
import myfilter
import mydaq
import myClassifier

data_1d_original: None = None
data_1d_processed = None

data_2d_original: None = None
data_2d_processed = None

image_lang: None = None

def show_main_menu():
    print("menu")
    print("1. Read/Write Data")
    print("2. Apply Filters")
    print("3. Extract Features")
    print("4. Pore Classification")
    print("5. Detect Defects")
    print("6. Language Detection")
    while True:
        try:
            choice = int(input("Enter your Selection"))
            process_main_menu(choice)
        except ValueError:
            print("[Error] Please enter a valid option.")
            continue


def process_main_menu(choice):
    if choice == 1:
        show_sub_menu1()
    if choice == 2:
        show_sub_menu2()
    if choice == 3:
        show_sub_menu3()
    if choice == 4:
        show_sub_menu4()
    if choice == 5:
        show_sub_menu5()
    if choice == 6:
        show_sub_menu6()
    else:
        print("Invalid selection!")
        show_main_menu()


def show_sub_menu1():
    print("Read /Write Data:")
    print("1. Read and Plot Data from a CSV File")
    print("2. Write Data to a CSV File")
    print("3. Read and Plot Data from an Image")
    print("4. Write Data to an Image")
    print("5. Acquire Data From Arduino")
    print("-1. Back to the Main Menu")
    while True:
        try:
            choice = int(input("Enter your selection:"))
            process_sub_menu1(choice)
        except ValueError:
            print("[Error] Please Enter a Valid Option.")
            continue


def process_sub_menu1(choice):
    if choice == 1:
        print("[Start] Read and plot data from a csv file")
        global data_1d_original
        data_1d_original = myfile.read_csv_file()
        myplot.plot_1d_data(data_1d_original)
        print("[End] Read and plot data from a csv file")
        show_main_menu()
    elif choice == 2:
        print("[Start] Write data to a csv file")
        global data_1d_processed
        myfile.write_csv_file(data_1d_processed)
        myplot.plot_1d_data(data_1d_processed)
        print("[End] Write data to a csv file")
        show_main_menu()
    elif choice == 3:
        print("[Start] Read and plot data from an image")
        global data_2d_original
        data_2d_original = myfile.read_image_file()
        myplot.show_image(data_2d_original)
        print("[End] Read and plot data from an image")
        show_main_menu()
    elif choice == 4:
        global data_2d_processed
        print("[start] Write data to an image")
        global data_2d_processed
        myfile.write_image_file(data_2d_processed)
        print("[End] write data to an image")
        show_main_menu()
    elif choice == 5:
        data_1d_original = mydaq.acquire_arduino_data()
        myplot.plot_1d_data(data_1d_original)
        mydaq.save_to_csv(data_1d_original)
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu1()


def show_sub_menu2():
    print("Apply Filter:")
    print("1. Apply Averaging Filter")
    print("2. Apply Magnitude Filter")
    print("3. Apply Gaussian Filter On Image")
    print("4. Apply Sobel Filter On Image")
    print("5. Apply Canny Filter On Image")
    print("6. Clean Data and Extract New Features")
    print("7. Remove Pattern")
    print("-1. Back to the Main Menu")
    while True:
        try:
            choice = int(input("Enter your selection:"))
            process_sub_menu2(choice)
        except ValueError:
            print("[Error] Please Enter a Valid Option.")
            continue


def process_sub_menu2(choice):
    global data_1d_original
    global data_1d_processed
    global data_2d_original
    global data_2d_processed

    if choice == 1:
        print("[Start] Apply averaging filter")
        data_1d_processed = myfilter.gaussian_filter(data_1d_original, 1.5)
        myplot.plot_two_1d_data(data_1d_original, data_1d_processed)
        print("[End] Apply averaging filter")
        show_main_menu()
    elif choice == 2:
        print("[Start] Apply magnitude filter")
        data_1d_processed = myfilter.magnitude_filter(data_1d_original, 7, 8)
        myplot.plot_1d_data(data_1d_processed)
        print("[End] Apply magnitude filter")
    elif choice == 3:
        print("[Start] Apply Gaussian filter on image")
        data_2d_processed = myfilter.gaussian_filter_img(data_2d_original)
        myplot.show_two_images(data_2d_original, data_2d_processed)
        print("[End] Apply Gaussian filter on image")
        show_main_menu()
    elif choice == 4:
        print("[Start] Apply Sobel filter on image")
        data_2d_processed = myfilter.sobel_filter_img(data_2d_original)
        myplot.show_two_images(data_2d_original, data_2d_processed, sobel=True)
        print("[End] Apply Sobel filter on image")
        show_main_menu()
    elif choice == 5:
        print("[Start] Apply Canny filter on image")
        data_2d_processed = myfilter.canny_filter_img(data_2d_original)
        myplot.show_two_images(data_2d_original, data_2d_processed)
        print("[End] Apply Sobel filter on image")
        show_main_menu()
    elif choice == 6:
        print("[Start] Clean Data and Extract New Features")
        data_1d_processed = myfilter.clean_data(data_1d_original)
        myplot.plot_1d_data(data_1d_processed)
        print('Min = ', myFeature.calc_min(data_1d_processed))
        print('Max = ', myFeature.calc_max(data_1d_processed))
        print('Avg = ', myFeature.calc_ave(data_1d_processed))
        print('Std = ', myFeature.calc_std(data_1d_processed))
        print('Spectral Entropy = ', myFeature.calc_spectral_entropy(data_1d_processed))
        print("[End] Clean Data and Extract New Features")
        show_main_menu()
    elif choice == 7:
        print("[Start] Remove Pattern")
        data_2d_processed = myfilter.remove_pattern(data_2d_original, 20, -20)
        myplot.show_image(data_2d_processed)
        print("[End] Remove Pattern")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu2()


def show_sub_menu3():
    print("Extract Features:")
    print("1. All Features")
    print("2. Average")
    print("3. Apply Fourier Transform")
    print("4. Classify Features")
    print("-1. Back to the Main Menu")
    while True:
        try:
            choice = int(input("Enter your selection:"))
            process_sub_menu3(choice)
        except ValueError:
            print("[Error] Please Enter a Valid Option.")
            continue


def process_sub_menu3(choice):
    global data_1d_original
    global data_1d_processed

    if choice == 1:
        print('Min = ', myFeature.calc_min(data_1d_original))
        print('Max = ', myFeature.calc_max(data_1d_original))
        print('Avg = ', myFeature.calc_ave(data_1d_original))
        print('Std = ', myFeature.calc_std(data_1d_original))
        print('Spectral Entropy = ', myFeature.calc_spectral_entropy(data_1d_original))
        show_main_menu()
    elif choice == 2:
        print('Average = ', myFeature.calc_ave(data_1d_original))
        show_main_menu()
    elif choice == 3:
        print("[Start] Apply Fourier Transform")
        myFeature.apply_fft(data_1d_original)
        print("[End] Apply Fourier Transform")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu3()


def show_sub_menu4():
    print("Pore Classification:")
    print("1. Apply Vector Machine Method")
    print("2. Apply Random Forest Method")
    print("3. Apply Decision Tree method")
    print("4. Apply CNN-based method")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection:"))
            process_sub_menu4(choice)
        except ValueError:
            print("[Error] Please Enter a Valid Option.")
            continue


def process_sub_menu4(choice):
    if choice == 1:
        print("[Start] Apply Vector Machine method")
        myClassifier.classify('svm')
        print("[End] Apply Vector Machine method")
        show_main_menu()
    elif choice == 2:
        print("[Start] Apply Random Forest method")
        myClassifier.classify('rf')
        print("[End] Apply Random Forest method")
        show_main_menu()
    elif choice == 3:
        print("[Start] Apply Decision Tree method")
        myClassifier.classify('dt')
        print("[End] Apply Decision Tree method")
        show_main_menu()
    elif choice == 4:
        print("[Start] Apply CNN-based method")
        myClassifier.classify('cnn')
        print("[End] Apply CNN-based method")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu4()


def show_sub_menu5():
    print("1. LCD Defect Detection Using Pattern Comparison")
    print("2. LCD Defect Detection Using FFT")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection:"))
            process_sub_menu5(choice)
        except ValueError:
            print("[Error] Please Enter a Valid Option.")
            continue


def process_sub_menu5(choice):
    global data_2d_original
    global data_2d_processed
    if choice == 1:
        pattern_period = myfilter.find_pattern_period(data_2d_original)
        myfilter.eliminate_pattern(data_2d_original, pattern_period)
        show_main_menu()
    elif choice == 2:
        data_2d_processed = myfilter.detect_defects_fft(data_2d_original)
        myplot.show_complex_image(data_2d_processed, "FFT Defect Detection")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu5()


def show_sub_menu6():
    print("1. Select Image File")
    print("2. Print Text")
    print("3. Test Classifier")
    print("4. Detect Language From Image Using CNN")
    print("-1. Back to the main menu")
    while True:
        try:
            choice = int(input("Enter your selection:"))
            process_sub_menu6(choice)
        except ValueError:
            print("[Error] Please Enter a Valid Option.")
            continue


def process_sub_menu6(choice):
    if choice == 1:
        print("[Start] Select Image File")
        global image_lang
        image_lang = myLanguageDetection.select_image()
        myplot.show_image(image_lang)
        print("[End] Select Image File")
    elif choice == 2:
        print("[Start] Print Text")
        text = myLanguageDetection.ocr_result(image_lang)
        print(text)
        print("[End] Print Text")
        show_main_menu()
    elif choice == 3:
        print("[Start] Test Classifier")
        myLanguageDetection.training_testing_dt()
        print("[End] Test Classifier")
        show_main_menu()
    elif choice ==4:
        print("[Start] Detect Language From Image Using CNN")
        myLanguageDetection.detect_language_cnn(image_lang)
        print("[Start] Detect Language From Image Using CNN")
        show_main_menu()
    elif choice == -1:
        show_main_menu()
    else:
        print("Invalid selection!")
        show_sub_menu6()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    show_main_menu()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
