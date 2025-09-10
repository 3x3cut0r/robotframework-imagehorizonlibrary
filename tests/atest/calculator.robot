*** Settings ***
Library    OperatingSystem

*** Test cases ***

Calculator
    Set Test Variable    ${LIB_IMPORTED}    False
    ${run_gui}=    Get Environment Variable    RUN_GUI_TESTS    0
    Skip If    '${run_gui}' != '1'    GUI or browser not available
    Import Library    ImageHorizonLibrary    ${CURDIR}${/}reference_images${/}calculator    screenshot_folder=${OUTPUT_DIR}
    Set Test Variable    ${LIB_IMPORTED}    True
    Set Confidence      0.9
    Launch application    python3 tests/atest/calculator/calculator.py
    ${location1}=    Wait for    inputs_folder     timeout=30
    Click to the above of     ${location1}    20
    Type    1010
    Click to the below of     ${location1}    20
    Type    1001
    ${location2}=    Locate    or_button.png
    Click to the below of     ${location2}    0
    Click to the below of     ${location2}    50
    ${result}=    Copy
    Should be equal as integers    ${result}    1011
    Click Image     close_button.png
    [Teardown]    Run Keyword If    ${LIB_IMPORTED}    Terminate application
