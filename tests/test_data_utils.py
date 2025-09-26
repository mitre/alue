import pytest
import os

from alue.data_utils import (
    load_task_data,
    DataLoader,
)


@pytest.fixture(scope="module")
def current_dir():
    return os.path.dirname(__file__)


def test_asrs_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "ASRS_rag/rag_qa.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 4)
    assert(examples[0]['input'] == "What is the location of the airport that needs its grass mowed?")
    assert(examples[0]['output'] == "SC")
    assert(examples[0]['context'].startswith("ACN: 1834092"))
    assert(len(test_data) == 8)
    assert(test_data[0]['input'] == "What is the location of the airport that needs its grass mowed?")
    assert(test_data[0]['output'] == "SC")


def test_aviation_knowledge_exam_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "aviation_knowledge_exam/3_1_aviation_test.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 5)
    assert(examples[0]['input'] == "What part of a rotary-wing aircraft makes directional control possible?\r\nA) the teeter hinge\r\nB) the swashplate\r\nC) the ducted fan")
    assert(examples[0]['output'] == "B")
    assert(len(test_data) == 238)
    assert(test_data[0]['input'] == (
        "What should a pilot do if after crossing a stop bar, the taxiway centerline lead-on lights inadvertently extinguish?\r\nA) "
        "Proceed with caution\r\nB) Hold their position and contact ATC\r\nC) Turn back towards the stop bar"
    ))
    assert(test_data[0]['output'] == "B")


def test_dummy_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "dummy_rag/rag_qa.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 2)
    assert(examples[0]['input'] == "What is the purpose of FAA Order 8040.1C?")
    assert(examples[0]['context'] == (
        "This order describes the Federal Aviation Administration's (FAA) authority and assigns "
        "responsibility for the development and issuance of Airworthiness Directives (AD) in "
        "accordance with applicable statutes and regulations."
    ))
    assert(examples[0]['output'] == (
        "To describe the FAA's authority and assign responsibility for developing "
        "and issuing Airworthiness Directives in accordance with applicable statutes and regulations."
    ))
    assert(len(test_data) == 2)
    assert(test_data[0]['input'] == "What is the effective date of FAA Order 8040.1C on Airworthiness Directives?")
    assert(test_data[0]['output'] == "10/03/07")


def test_nasa_nodis_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "NASA_NODIS_rag/rag_qa.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 2)
    assert(examples[0]['input'] == "What strategies can NASA employ to enhance cybersecurity across its operations?")
    assert(examples[0]['context'] == (
        "To enhance cybersecurity across its operations, NASA can implement several strategies. These include "
        "adopting a zero-trust architecture, which assumes that threats could be internal or external and requires "
        "strict identity verification for every person and device trying to access resources. Regular security audits "
        "and vulnerability assessments can help identify and mitigate potential risks. Employee training programs "
        "focused on cybersecurity awareness can reduce the likelihood of human error leading to security breaches. "
        "Additionally, NASA can leverage advanced technologies such as artificial intelligence and machine learning "
        "to detect and respond to threats in real-time. Collaboration with other federal agencies and industry "
        "partners can also provide valuable insights and resources for strengthening cybersecurity measures."
    ))
    assert(examples[0]['output'] == (
        "Adopt zero-trust architecture, conduct regular audits, train employees, use AI/ML for threat detection, "
        "and collaborate with partners."
    ))
    assert(len(test_data) == 18)
    assert(test_data[0]['input'] == (
        "Who has has a statutory mandate to promote economy, efficiency, and effectiveness in the "
        "administration of NASA\nprograms and operations and to prevent and detect crime, fraud, "
        "waste, abuse, and mismanagement in such\nprograms and operations?"
    ))
    assert(test_data[0]['output'] == "The Inspector General (IG) ")


def test_nasa_standards_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "NASA_Standards_rag/rag_qa.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 2)
    assert(examples[0]['input'] == "What are the key components of a robust system engineering process?")
    assert(examples[0]['context'] == (
        "A robust system engineering process involves several key components to ensure the successful "
        "development and deployment of complex systems. These components include requirements analysis, "
        "where system needs are clearly defined and documented. Design synthesis follows, where solutions "
        "are developed to meet these requirements. Integration and testing are critical to verify that "
        "the system functions as intended. Risk management is also essential, involving the identification "
        "and mitigation of potential issues that could impact project success. Finally, continuous "
        "evaluation and feedback loops help refine the process and adapt to changing conditions or new "
        "information."
    ))
    assert(examples[0]['output'] == (
        "Requirements analysis, design synthesis, integration and testing, risk management, and "
        "continuous evaluation."
    ))
    assert(len(test_data) == 10)
    assert(test_data[0]['input'] == "What are the three aspects of MBSE?")
    assert(test_data[0]['output'] == "MBSE has three aspects: the modeling language, the modeling methodology, and the modeling")


def test_nasa_systems_engineering_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "NASA_Systems_Engineering_rag/rag_qa.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 2)
    assert(examples[0]['input'] == "What are the benefits of using a model-based systems engineering (MBSE) approach?")
    assert(examples[0]['context'] == (
        "Model-based systems engineering (MBSE) offers several benefits that enhance the development and management "
        "of complex systems. By using models as the primary means of information exchange, MBSE improves communication "
        "among stakeholders and reduces ambiguity in requirements. It facilitates early detection of design errors and "
        "inconsistencies, allowing for timely corrections. MBSE also supports traceability and impact analysis, making "
        "it easier to manage changes throughout the system lifecycle. Additionally, the use of standardized modeling "
        "languages and tools enables better integration and interoperability across different teams and systems."
    ))
    assert(examples[0]['output'] == (
        "Improves communication, reduces ambiguity, facilitates error detection, supports traceability, "
        "and enables integration."
    ))
    assert(len(test_data) == 4)
    assert(test_data[0]['input'] == "outline Product Realization Keys for systems engineering")
    assert(test_data[0]['output'] == (
        "Product Realization Keys\nDefine and execute production activities.\nGenerate and manage requirements for "
        "off-the-shelf\nhardware/software products as for all other products.\nUnderstand the differences between "
        "verification testing and\nvalidation testing.\nConsider all customer, stakeholder, technical, programmatic, "
        "and\nsafety requirements when evaluating the input necessary to\nachieve a successful product transition.\n"
        "Analyze for any potential incompatibilities with interfaces as\nearly as possible.\nCompletely understand "
        "and analyze all test data for trends and\nanomalies.\nUnderstand the limitations of the testing and any "
        "assumptions\nthat are made.\nEnsure that a reused product meets the verification and\nvalidation required "
        "for the relevant system in which it is to be\nused, as opposed to relying on the original verification and\n"
        "validation it met for the system of its original use. Then ensure\nthat it meets the same verification and "
        "validation as a purchased\nproduct or a built product. The ÒpedigreeÓ of a reused product in\nits original "
        "application should not be relied upon in a different\nsystem, subsystem, or application."
    ))


def test_notam_classification_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "NOTAM_binary_classification/2_3_classification_test.json")
    loader = DataLoader(data_path)
    test_data = loader.get_test_data()

    assert(len(test_data) == 100)
    assert(test_data[0]['input'] == "PVU SVC TWR CLSD")
    assert(test_data[0]['output'] == "Yes")


def test_ntml_sentiment_analysis_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "NTML_sentiment_analysis/2_2_NTML_SA_v1.json")
    loader = DataLoader(data_path)
    test_data = loader.get_test_data()

    assert(len(test_data) == 88)
    assert(test_data[0]['input'] == (
        "ANOTHER VERY IMPACTFUL DAY FOR DELTA. THE AFPJX5 RATES APPEARED TO BE APPROPRIATE IN THE "
        "INITIAL RUN BUT DID NOT PREVENT GROUND STOPS FOR NORTHBOUND FLIGHTS OUT OF ZMA. GETTING "
        "FLIGHTS OUT OF FLORIDA TO THE NORTH WAS VERY CHALLENGING."
    ))
    assert(test_data[0]['output'] == "negative")


def test_ntrs_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "NTRS_rag/rag_qa.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 2)
    assert(examples[0]['input'] == "What advancements have been made in the field of autonomous spacecraft navigation?")
    assert(examples[0]['context'] == (
        "Recent advancements in autonomous spacecraft navigation have significantly enhanced the ability of spacecraft "
        "to operate independently in space. These advancements include the development of sophisticated algorithms that "
        "enable real-time decision-making and path optimization. The integration of artificial intelligence and machine "
        "learning techniques allows spacecraft to adapt to changing conditions and unexpected obstacles. Additionally, "
        "improvements in sensor technology provide more accurate data for navigation systems, enhancing precision and "
        "reliability. These innovations are crucial for deep space missions, where communication delays with Earth make "
        "autonomous operation essential."
    ))
    assert(examples[0]['output'] == (
        "Development of sophisticated algorithms, integration of AI/ML, and improvements in sensor technology."
    ))
    assert(len(test_data) == 2)
    assert(test_data[0]['input'] == "what is an economical way to manufacture voxels?")
    assert(test_data[0]['output'] == (
        "injection molded chopped fiber composites offered\nan economical way to manufacture voxels that achieved "
        "performance regimes\nuseful for space structures"
    ))


def test_ntsb_damage_classification_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "ntsb_damage_classification/ntsb_damage_classification_v2.json")
    examples, test_data = load_task_data(data_path)

    print(examples[0])
    print(test_data[0])

    assert(len(examples) == 5)
    assert(examples[0]['input'] == (
        "The pilot reported that during the night flight, and prior to landing, he discovered that both landing "
        "lights were burned out.  During the landing, the nose of the right skid contacted the ground and the "
        "helicopter bounced back into the air and rotated to the right resulting in the tail rotor striking a "
        "hangar.  The helicopter rotated about 360 degrees before impacting the ground.  The helicopter sustained "
        "substantial damage to the tail rotor gearbox attachment point.  The pilot reported no other abnormalities "
        "with the helicopter prior to the accident."
    ))
    assert(examples[0]['output'] == "SUBS")
    assert(len(test_data) == 1994)
    assert(test_data[0]['input'] == (
        "The pilot reported while making an approach to a hover about 15 feet above ground level (agl), he "
        "applied power to stop the decent rate and the helicopter began to yaw to the right.  Despite the "
        "pilot adding left anti-torque pedal and increasing power, the helicopter continued to yaw to the "
        "right and ascended 50-75 feet.  The pilot stated he lowered the collective and reduced power until "
        "the helicopter descended through about 25 feet agl, and then he raised the collective for landing.  "
        "Subsequently, the helicopter landed hard within sandy terrain on the shoreline of a lake.  Examination "
        "of the helicopter revealed that the tail boom and firewall sustained substantial damage.  No "
        "mechanical anomalies were noted during the examination."
    ))
    assert(test_data[0]['output'] == "SUBS")


def test_ntsb_tail_extraction_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "ntsb_tail_extraction/ntsb_extractive_qa.json")
    examples, test_data = load_task_data(data_path)

    print(examples[0])
    print(test_data[0])

    assert(examples[0]['question'] == (
        "Is a tail number mentioned in this transcript? If so, output the response as an array "
        "containing the tail number(s). Example: [N44NV, N8280J]. Output an array containing NONE "
        "if no tail number is mentioned in the transcript or if you do not know the answer to the "
        "question. Do not output any aircraft callsigns mentioned in the transcript"
    ))
    assert(examples[0]['transcript'] == (
        "On March 18, 2008 approximately 1045 mountain daylight time, a Robinson R22 beta, M3056T, "
        "registered to Pacific Rim Aviation, Lafayette, Colorado, and operated by Premier Helicopters, "
        "Broomfield, Colorado, was substantially damaged when it impacted terrain following a loss of "
        "rotor rpm during a practice autorotation at Las Vegas Municipal Airport (LVS), Las Vegas, "
        "New Mexico. Visual meteorological conditions (VMC) prevailed at the time of the accident. "
        "The cross-country instructional flight was being conducted under the provisions of Title 14 "
        "Code of Federal Regulations (CFR) Part 91 without a flight plan. The commercial certificated "
        "flight instructor and student pilot on board the helicopter were not injured. The flight "
        "originated at Raton, New Mexico, at 0900, and was en route to Las Vegas, New Mexico.\n\n"
        "According to the instructor's accident report, they checked both the terminal forecast (TAF) "
        "and the Automated Weather Observation Station (AWOS) for LVS. Both reported the wind to be "
        "from 020 degrees, but the TAF reported wind velocity to be 10 knots, whereas AWOS reported it "
        "to be 15 knots with gusts to 20 knots. As the helicopter approached the airport, the student "
        "told the instructor he would like to practice a 180 degree autorotation. The maneuver was "
        "begun from 1,000 feet agl (above ground level), but due to poor rpm and speed control, the "
        "instructor assumed control of the helicopter and advised the student he would demonstrate the "
        "proper technique. The instructor said he kept the rpm 'in the green' and kept the airspeed "
        "above 65 knots, but realized the helicopter was descending 'at an unusually high rate of "
        "descent.' As he rolled out to level, 'it felt as if we had a huge downdraft that was pushing "
        "us towards the ground.' The instructor added throttle and began to 'pull all the power "
        "available.' Lift was not sufficient to overcome the descent rate. Just prior to impact, the "
        "pilot applied slight aft cyclic control. The helicopter struck the ground, bounced, and "
        "continued to fly with some forward momentum. The instructor then landed the helicopter. "
        "Post-impact inspection revealed the skids were spread, the engine mounts were bent, and part "
        "of the pilot's window had popped out. The instructor said that at no time did the LOW ROTOR "
        "RPM horn sound."
    ))
    assert(examples[0]['answer'] == "[M3056T]")
    assert(len(test_data) == 1844)
    assert(test_data[0]['input'] == (
        "Is a tail number mentioned in this transcript? If so, output the response as an array "
        "containing the tail number(s). Example: [N44NV, N8280J]. Output an array containing NONE "
        "if no tail number is mentioned in the transcript or if you do not know the answer to the "
        "question. Do not output any aircraft callsigns mentioned in the transcript"
    ))
    assert(test_data[0]['output'] == "['NONE']")


def test_site_licenses_rag_data_loader(current_dir):

    data_path = os.path.join(current_dir, "../data/", "Site_Licenses_rag/Site Licenses.json")
    examples, test_data = load_task_data(data_path)

    assert(len(examples) == 2)
    assert(examples[0]['input'] == "What considerations are important for selecting a launch site?")
    assert(examples[0]['context'] == (
        "When selecting a launch site, several considerations are crucial to ensure safety and operational "
        "efficiency. These include geographical location, which affects the trajectory and potential impact "
        "zones; proximity to populated areas, which must be minimized to reduce risk; and environmental factors, "
        "such as weather patterns and natural hazards. Additionally, logistical support, including access to "
        "transportation and communication infrastructure, is essential. Regulatory compliance with local, state, "
        "and federal laws is also a key factor in the site selection process."
    ))
    assert(examples[0]['output'] == (
        "Geographical location, proximity to populated areas, environmental factors, logistical support, and "
        "regulatory compliance."
    ))
    assert(len(test_data) == 4)
    assert(test_data[0]['input'] == (
        " What is the purpose of the redesignation in the Commercial Space Launch Act as stated in the document "
        "issued in June 23, 2015?"
    ))
    assert(test_data[0]['output'] == (
        'Due to the recodification of the Commercial Space Launch Act in the federal\ncode, redesignated '
        'Authority to read: "51 U.S.C. Subtitle V, Ch. 509.'
    ))

def test_summarization_data_loader(current_dir):
    """Test the new JSONL format with split field for summarization task."""
    
    data_path = os.path.join(current_dir, "../data/", "asrs_summarization/asrs_summarization.jsonl")
    examples, test_data = load_task_data(data_path)

    # Test examples
    assert len(examples) > 0, "Should have examples from the JSONL file"
    assert "input" in examples[0], "Examples should have 'input' field"
    assert "output" in examples[0], "Examples should have 'output' field" 
    
    # Verify it's actually loading from the "example" split
    assert examples[0]['input'].startswith("After takeoff"), "Should match expected example content"
    
    # Test test data
    assert len(test_data) > 0, "Should have test data from the JSONL file"
    assert "input" in test_data[0], "Test data should have 'input' field"
    assert "output" in test_data[0], "Test data should have 'output' field"
    
    # Verify examples and test data are different
    assert len(examples) != len(test_data), "Examples and test data should have different lengths"
    
    print(f"✓ Loaded {len(examples)} examples and {len(test_data)} test items from JSONL")