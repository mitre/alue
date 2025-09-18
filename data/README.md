# Datasets README

## Tasks

### Multiple Choice Question Answering

#### Aviation Knowledge Exam MCQA

This task involves evaluating the ability of LLMs to comprehend aviation knowledge and to emulate aviation professionals in resolving real-world aviation issues. The dataset was curated using data from aviation websites and books such as the Airplane Flying Handbook (FAA-H-8083-3C), Aeronautical Information Manual, Helicopter Flying Handbook, Instrument Flying Handbook, Pilot's Handbook of Aeronautical Knowledge (FAA-H-8083-25C), etc.

The **Aviation Knowledge Exam** contains 238 aviation knowledge exam questions in a multiple-choice format. Each sample includes:

- `question`: The question text, followed by answer options labeled (A), (B), (C), etc.
- `answer`: The correct option, indicated by its letter (e.g., "B").
- `id` (for some samples): Unique identifier for the question

### Example



```json
{
  "question": "What should a pilot do if after crossing a stop bar, the taxiway centerline lead-on lights inadvertently extinguish?\r\nA) Proceed with caution\r\nB) Hold their position and contact ATC\r\nC) Turn back towards the stop bar",
  "answer": "B",
  "id": 0
}
```

### Classification

#### NOTAM Tower Closure Classification

NOTAMs are critical communications issued by the Federal Aviation Administration to inform pilots and other personnel about essential information that could affect flight operations. This includes updates on airport conditions, airspace restrictions, equipment outages, and other safety-related details. NOTAMs are crucial for ensuring that pilots have the most current information for safe and efficient flight planning and execution.

This binary classification task involves classifying a NOTAM based on whether it contains a notice on a closed tower.

There are a total of 100 samples in the dataset

Each sample includes:

- `text`: The NOTAM message
- `label`: "Yes" if the message indicates a tower closure, "No" otherwise

### Example

```json
{
  "text": "PVU SVC TWR CLSD",
  "label": "Yes"
}
```

```json
{
  "text": "CANCELED\"",
  "label": "No"
}
```


#### National Traffic Management Log (NTML) Sentiment Classification

The National Traffic Management Log (NTML) is a tool used by the Federal Aviation Administration to document and share information about traffic management initiatives and decisions. The NTML helps coordinate and communicate among various air traffic control facilities to ensure efficient and safe management of air traffic across the national airspace system.

This task involves assessing sentiments expressed by airline officials in response to Traffic Management Initiatives published in NTML. The sentiments are either expressed as bad, good, or neutral.

There are a total of 88 samples in the dataset.

Each sample includes:

- `text`: The comment or message from the NTML
- `label`: The sentiment classification ("positive", "negative", or "neutral")

### Example

```json
{
  "text": "QUIET EVENING SHIFT, THANKS FOR YOUR HELP TODAY.",
  "label": "positive"
}
```

```json
{
  "text": "FRUSTRATING DAY.",
  "label": "negative"
}
```

```json
{
  "text": "NO ISSUES.",
  "label": "neutral"
}
```


#### NTSB Tail Aircraft Damage Classification Dataset

NTSB reports can describe the level of damage sustained by aircraft in an accident. The damage level can be classified as Substantial (SUBS), Minor (MINR), Destroyed (DEST), or None (NONE). This task is a multi-class classification of aircraft damage reported in NTSB reports.

The NTSB Tail Aircraft Damage Classification dataset consists of 1994 samples derived from various NTSB reports. This dataset is formatted for sequence classification.

Each sample includes:

- `Input`: The accident report text
- `Output` or `labels`: The damage classification ("SUBS", "MINR", "DEST", or "NONE")

### Example

```json
{
  "Input": [
    "The pilot reported that during the night flight, and prior to landing, he discovered that both landing lights were burned out. During the landing, the nose of the right skid contacted the ground and the helicopter bounced back into the air and rotated to the right resulting in the tail rotor striking a hangar. The helicopter rotated about 360 degrees before impacting the ground. The helicopter sustained substantial damage to the tail rotor gearbox attachment point. The pilot reported no other abnormalities with the helicopter prior to the accident."
  ],
  "Output": "SUBS"
}
```

```json
{
  "text_input": [
    "The pilot reported while making an approach to a hover about 15 feet above ground level (agl), he applied power to stop the decent rate and the helicopter began to yaw to the right. Despite the pilot adding left anti-torque pedal and increasing power, the helicopter continued to yaw to the right and ascended 50-75 feet. The pilot stated he lowered the collective and reduced power until the helicopter descended through about 25 feet agl, and then he raised the collective for landing. Subsequently, the helicopter landed hard within sandy terrain on the shoreline of a lake. Examination of the helicopter revealed that the tail boom and firewall sustained substantial damage. No mechanical anomalies were noted during the examination."
  ],
  "labels": "SUBS"
}
```


### Extractive Question Answering

#### NTSB Tail Number Entity Extraction

NTSB reports can contain the tail number(s) of aircraft associated with an aviation accident. This task focuses on extraction of tail numbers reported in NTSB reports. The outputs of the models is an array of tail numbers contained in a transcript in the form of ['N1234','N6789'].

The NTSB Tail Number Entity Extraction dataset consists of 1844 examples derived from various NTSB transcripts. This dataset is formatted in the SQuAD (Stanford Question Answering Dataset) format for extractive question answering.

Each sample includes:

- `context`: The transcript text from the NTSB report
- `qas`:
  - `question`: The extraction prompt
  - `answers`:
    - `text`: The extracted tail number(s) as an array (e.g., `['N66372']`)

### Example

```json
{
  "context": "HISTORY OF FLIGHT\n\nOn February 9, 2008, at 1115 central standard time, a Hughes OH-6A single-engine helicopter, N66372, was substantially damaged during a forced landing to a field after a loss of engine power near Valentine, Texas. The commercial pilot sustained serious injuries. The helicopter was registered to the United States Border Patrol Air Operations and operated by Customs and Border Protection (CBP). No flight plan was filed for the flight that departed Marfa Municipal Airport (MRF), near Marfa, Texas, about 1027. ...",
  "qas": [
    {
      "id": "d247a3e7-8e7a-4389-9dc2-3be048fef873",
      "question": "Is a tail number mentioned in this transcript? If so, output the response as an array containing the tail number(s). Example: [N44NV, N8280J]. Output an array containing NONE if no tail number is mentioned in the transcript or if you do not know the answer to the question. Do not output any aircraft callsigns mentioned in the transcript",
      "answers": [
        {
          "text": "['N66372']"
        }
      ]
    }
  ]
}
```

### Retrieval Augmented Generation (RAG)

#### Aviation Safety Reporting System (ASRS) RAG


The **ASRS RAG Dataset** contains 8 question-answer samples from aviation safety reports. Each sample includes:

- `id`: Unique identifier
- `question`: The question about an incident
- `answers`:
  - `text`: Ground truth answer
  - `document_id`: List of source document accession numbers

### Example

```json
{
  "id": "2",
  "question": "What is the location of the airport that needs its grass mowed?",
  "answers": [
    {
      "text": "SC",
      "document_id": ["1834092"]
    }
  ]
}
```

---

#### NASA NODIS RAG

The **NASA NODIS Dataset** contains 18 question-answer samples based on NASA policy and directive documents. Each sample includes:

- `id`: Unique identifier
- `question`: The question about NASA policies or procedures
- `answers`:
  - `text`: Ground truth answer
  - `document_id`: List of source document IDs

### Example

```json
{
  "id": "0",
  "question": "Who has a statutory mandate to promote economy, efficiency, and effectiveness in the administration of NASA programs and operations and to prevent and detect crime, fraud, waste, abuse, and mismanagement in such programs and operations?",
  "answers": [
    {
      "text": "The Inspector General (IG)",
      "document_id": ["bd19619b1e289651aa074a8226fcf1a6f814a940c34a02d7b9e8084c15a7dc8f"]
    }
  ]
}
```

---

#### NASA Standards RAG


The **NASA Standards Dataset** contains 10 question-answer samples based on NASA engineering and safety standards. Each sample includes:

- `id`: Unique identifier
- `question`: The question about NASA standards or practices
- `answers`:
  - `text`: Ground truth answer
  - `document_id`: List of source document IDs

### Example

```json
{
  "id": "0",
  "question": "What are the three aspects of MBSE?",
  "answers": [
    {
      "text": "MBSE has three aspects: the modeling language, the modeling methodology, and the modeling",
      "document_id": ["e360ef9e44523bb8679ef962ff1a175e831875668f52fb976d787996ab4ea64d"]
    }
  ]
}
```

#### NASA Systems Engineering RAG

The **NASA Systems Engineering Dataset** contains 4 question-answer samples based on the NASA Systems Engineering Handbook. Each sample includes:

- `id`: Unique identifier
- `question`: The question about systems engineering concepts or practices
- `answers`:
  - `text`: Ground truth answer
  - `document_id`: List of source document IDs

### Example

```json
{
  "id": "0",
  "question": "outline Product Realization Keys for systems engineering",
  "answers": [
    {
      "text": "Product Realization Keys\nDefine and execute production activities.\nGenerate and manage requirements for off-the-shelf hardware/software products as for all other products.\nUnderstand the differences between verification testing and validation testing.\nConsider all customer, stakeholder, technical, programmatic, and safety requirements when evaluating the input necessary to achieve a successful product transition.\nAnalyze for any potential incompatibilities with interfaces as early as possible.\nCompletely understand and analyze all test data for trends and anomalies.\nUnderstand the limitations of the testing and any assumptions that are made.\nEnsure that a reused product meets the verification and validation required for the relevant system in which it is to be used, as opposed to relying on the original verification and validation it met for the system of its original use. Then ensure that it meets the same verification and validation as a purchased product or a built product. The “pedigree” of a reused product in its original application should not be relied upon in a different system, subsystem, or application.",
      "document_id": ["091555a35adc6a49be6f9925373715f8cb0c6cf579b14273d375ab294d18c111"]
    }
  ]
}
```

#### NASA Technical Reports Server (NTRS) RAG

The **NTRS Dataset** contains 2 question-answer samples based on NASA technical reports. Each sample includes:

- `id`: Unique identifier
- `question`: The question about technical advancements or research findings
- `answers`:
  - `text`: Ground truth answer
  - `document_id`: List of source document IDs

### Example

```json
{
  "id": "0",
  "question": "what is an economical way to manufacture voxels?",
  "answers": [
    {
      "text": "injection molded chopped fiber composites offered an economical way to manufacture voxels that achieved performance regimes useful for space structures",
      "document_id": ["f22d5d48c41e8f27bfe3c8e92630db56a4972f891c2b5379b0f9ad669cbf75eb"]
    }
  ]
}
```


#### Site Licenses RAG

The **Site Licenses Dataset** contains 4 question-answer samples based on commercial space launch site licensing documents. Each sample includes:

- `id`: Unique identifier
- `question`: The question about site licensing requirements or regulations
- `answers`:
  - `text`: Ground truth answer
  - `document_id`: List of source document IDs

### Example

```json
{
  "id": "0",
  "question": "What is the purpose of the redesignation in the Commercial Space Launch Act as stated in the document issued in June 23, 2015?",
  "answers": [
    {
      "text": "Due to the recodification of the Commercial Space Launch Act in the federal code, redesignated Authority to read: \"51 U.S.C. Subtitle V, Ch. 509.",
      "document_id": ["8ebcf100696d1724af8b9e031bc44551529aebfa8a6d73d63cb37ca13859561e"]
    }
  ]
}
```
