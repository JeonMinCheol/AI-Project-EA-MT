import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from DTOlist import TranslationDraft, EntityMemoryBlock, ERCMDecision
from vllm import LLM, SamplingParams
from decide_ercm import run_llm_judge, build_alias_set
test_cases = [
    {
        "name": "Normal 케이스",
        "source": "In what genre does A Promised Land fall?",
        "target_lang": "ar",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="A Promised Land",
            qid="Q101438737",
            canonical_target="الأرض الموعودة",
            alias_candidates=["الأرض الموعودة", "أه برومسد لاند"],
        ),
        "draft_text": "في أي نوع تقع الأرض الموعودة ؟",
    },
    {
        "name": "Omission 케이스",
        "source": "When was Tom and Jerry: The Movie released?",
        "target_lang": "ar",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Tom and Jerry: The Movie",
            qid="Q1090697",
            canonical_target="توم وجيري الفلم",
            alias_candidates=["توم وجيري الفلم"],
        ),
        "draft_text": "متى تم إصدار Tom and Jerry: The Movie؟",
    },
    {
        "name": "Residue 케이스",
        "source": "In what genre does A Promised Land fall?",
        "target_lang": "ar",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="A Promised Land",
            qid="Q101438737",
            canonical_target="الأرض الموعودة",
            alias_candidates=["الأرض الموعودة", "أه برومسد لاند"],
        ),
        "draft_text": "في أي نوع تقع الأرض الموعودة ؟가",
    },
    {
        "name": "Normal 케이스",
        "source": "Are soybean sprouts a common ingredient in Asian cuisine?",
        "target_lang": "ja",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="soybean sprout",
            qid="Q1036741",
            canonical_target="大豆もやし",
            alias_candidates=["大豆もやし", "豆もやし"],
        ),
        "draft_text": "豆もやしはアジア料理の一般的な材料ですか?",
    },
    {
        "name": "Grammar 케이스",
        "source": "What genre is Shut In?",
        "target_lang": "ja",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Shut In",
            qid="Q110968050",
            canonical_target="トジコメ",
            alias_candidates=["トジコメ"],
        ),
        "draft_text": "トジコメにジャンルは何ですか?",
    },
    {
        "name": "Residue 케이스",
        "source": "What genre is Shut In?",
        "target_lang": "ja",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Shut In",
            qid="Q110968050",
            canonical_target="トジコメ",
            alias_candidates=["トジコメ"],
        ),
        "draft_text": "トジコメにジャンルは무엇입니까?",
    },
    {
        "name": "Normal 케이스",
        "source": "What is Vezzolano Abbey known for?",
        "target_lang": "it",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Vezzolano Abbey",
            qid="Q1120660",
            canonical_target="abbazia di Vezzolano",
            alias_candidates=["abbazia di Vezzolano"],
        ),
        "draft_text": "Per cosa è famosa l'abbazia di Vezzolano?",
    },
    {
        "name": "Omission 케이스",
        "source": "In which year was the film Hello Stranger released?",
        "target_lang": "it",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Hello Stranger",
            qid="Q115749",
            canonical_target="Kuan meun ho",
            alias_candidates=["Kuan meun ho"],
        ),
        "draft_text": "In che anno è uscito il film ?",
    },
    {
        "name": "Residue 케이스",
        "source": "In which year was the film Hello Stranger released?",
        "target_lang": "it",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Hello Stranger",
            qid="Q115749",
            canonical_target="Kuan meun ho",
            alias_candidates=["Kuan meun ho"],
        ),
        "draft_text": "In che anno è uscito il film Kuan meun ho가?",
    },
    {
        "name": "Normal 케이스",
        "source": "Which country is the TV series Mid Morning Matters with Alan Partridge from?",
        "target_lang": "tr",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Mid Morning Matters with Alan Partridge",
            qid="Q14924035",
            canonical_target="Alan Partridge ile Mid Morning Matters",
            alias_candidates=["Alan Partridge ile Mid Morning Matters"],
        ),
        "draft_text": "Alan Partridge ile Mid Morning Matters dizisi hangi ülkenindir?",
    },
    {
        "name": "Omission 케이스",
        "source": "What type of artwork is inspired by The Royal Game?",
        "target_lang": "tr",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="The Royal Game",
            qid="Q1513820",
            canonical_target="Satranç",
            alias_candidates=["Satranç"],
        ),
        "draft_text": "adlı eserin yazarı kimdir?",
    },
    {
        "name": "Residue 케이스",
        "source": "Which country is the TV series Mid Morning Matters with Alan Partridge from?",
        "target_lang": "tr",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Mid Morning Matters with Alan Partridge",
            qid="Q14924035",
            canonical_target="Alan Partridge ile Mid Morning Matters",
            alias_candidates=["Alan Partridge ile Mid Morning Matters"],
        ),
        "draft_text": "Alan Partridge ile Mid Morning Matters dizisi hangi ülkenindir? 가",
    },
    {
        "name": "Normal 케이스",
        "source": "How would you describe the genre of the Violin Concerto?",
        "target_lang": "es",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Violin Concerto",
            qid="Q25786",
            canonical_target="Concierto para violín",
            alias_candidates=["Concierto para violín"],
        ),
        "draft_text": "¿Cómo describirías el género del Concierto para violín?",
    },
    {
        "name": "WrongAlias 케이스",
        "source": "How is el pepe remembered and worshipped as a god?",
        "target_lang": "es",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="The Royal Game",
            qid="Q29201",
            canonical_target="Huangdi",
            alias_candidates=["Huangdi", "Gelber Herrscher", "Gong Sun", "Xuan Yuan"],
        ),
        "draft_text": "¿Como se recuerda y adora a Gelber Herrscher como un dios?",
    },
    {
        "name": "아마도 Normal 케이스",
        "source": "How is el pepe remembered and worshipped as a god?",
        "target_lang": "es",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="The Royal Game",
            qid="Q29201",
            canonical_target="Huangdi",
            alias_candidates=["Huangdi", "Gelber Herrscher", "Gong Sun", "Xuan Yuan"],
        ),
        "draft_text": "¿Como se recuerda y adora a Huangdi como un dios?",
    },  
    {
        "name": "Residue 케이스",
        "source": "How is el pepe remembered and worshipped as a god?",
        "target_lang": "es",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="The Royal Game",
            qid="Q29201",
            canonical_target="Huangdi",
            alias_candidates=["Huangdi", "Gelber Herrscher", "Gong Sun", "Xuan Yuan"],
        ),
        "draft_text": "¿Como se recuerda y adora a Huangdi como un dios가?",
    },   
    {
        "name": "Normal 케이스",
        "source": "How many stations does the Dharma Initiative have?",
        "target_lang": "zh",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Dharma Initiative",
            qid="Q522336",
            canonical_target="達摩計劃",
            alias_candidates=["達摩計劃"],
        ),
        "draft_text": "達摩計劃有多少個站點？",
    },    
    {
        "name": "Omission 케이스",
        "source": "Who directed the 1984 film The Blood of Others?",
        "target_lang": "zh",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="The Blood of Others",
            qid="Q544607",
            canonical_target="情到深處無怨尤",
            alias_candidates=["情到深處無怨尤"],
        ),
        "draft_text": "誰導演了1984年的電影？",
    },
    {
        "name": "Residue 케이스",
        "source": "Who directed the 1984 film The Blood of Others?",
        "target_lang": "zh",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="The Blood of Others",
            qid="Q544607",
            canonical_target="情到深處無怨尤",
            alias_candidates=["情到深處無怨尤"],
        ),
        "draft_text": "誰導演了1984年的電影情到深處無怨尤가？",
    },
    {
        "name": "Normal 케이스",
        "source": "Is Father Brown a real or fictional entity?",
        "target_lang": "de",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Father Brown",
            qid="Q1545219",
            canonical_target="Pater Brown",
            alias_candidates=["Pater Brown", "Father Brown"],
        ),
        "draft_text": "Ist Pater Brown eine reale oder fiktive Person?",
    },
    {
        "name": "Omission 케이스",
        "source": "Where is Sakya Monastery located?",
        "target_lang": "de",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Sakya Monastery",
            qid="Q1585564",
            canonical_target="Sakya-Kloster",
            alias_candidates=["Sajia si", "Sagya-Kloster"],
        ),
        "draft_text": "Wo befindet sich das ?",
    },
    {
        "name": "Residue 케이스",
        "source": "Where is Sakya Monastery located?",
        "target_lang": "de",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Sakya Monastery",
            qid="Q1585564",
            canonical_target="Sakya-Kloster",
            alias_candidates=["Sajia si", "Sagya-Kloster"],
        ),
        "draft_text": "Wo befindet sich das Sakya-Kloster가?",
    },
    {
        "name": "Normal 케이스",
        "source": "How long was Callixtus III's papacy?",
        "target_lang": "th",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Father Brown",
            qid="Q160369",
            canonical_target="สมเด็จพระสันตะปาปากัลลิสตุสที่ 3",
            alias_candidates=["สมเด็จพระสันตะปาปากัลลิสตุสที่ 3"],
        ),
        "draft_text": "สมเด็จพระสันตะปาปากัลลิสตุสที่ 3 ดำรงตําแหน่งสันตะปาปานานแค่ไหน?",
    },
    {
        "name": "Omission 케이스",
        "source": "Is February 31 a real date in our calendar system or only existent in the fictional entity?",
        "target_lang": "th",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="February 31",
            qid="Q1648569",
            canonical_target="31 กุมภาพันธ์",
            alias_candidates=["31 กุมภาพันธ์"],
        ),
        "draft_text": "31 Şubatเป็นวันจริงในระบบปฏิทินของเราหรือมีอยู่ในสิ่งสมมติเท่านั้น?",
    },
    {
        "name": "Residue 케이스",
        "source": "How long was Callixtus III's papacy?",
        "target_lang": "th",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Father Brown",
            qid="Q160369",
            canonical_target="สมเด็จพระสันตะปาปากัลลิสตุสที่ 3",
            alias_candidates=["สมเด็จพระสันตะปาปากัลลิสตุสที่ 3"],
        ),
        "draft_text": "สมเด็จพระสันตะปาปากัลลิสตุสที่ 3 ดำรงตําแหน่งสันตะปาปานานแค่ไหน가?",
    },
    {
        "name": "Normal 케이스",
        "source": "How long is the Risou no Musuko TV series?",
        "target_lang": "ko",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Risou no Musuko",
            qid="Q1053156",
            canonical_target="이상의 아들",
            alias_candidates=["이상의 아들"],
        ),
        "draft_text": "이상의 아들 TV 시리즈의 방영 시간은 얼마입니까?",
    },
    {
        "name": "Grammer 케이스",
        "source": "Where is the Tomb of Hafez located?",
        "target_lang": "ko",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Tomb of Hafez",
            qid="Q129479",
            canonical_target="하페즈의 무덤",
            alias_candidates=["하페즈의 무덤"],
        ),
        "draft_text": "하페즈의 무덤는 어디에 있습니까?",
    },
    {
        "name": "Residue 케이스",
        "source": "Where is the Tomb of Hafez located?",
        "target_lang": "ko",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Tomb of Hafez",
            qid="Q129479",
            canonical_target="하페즈의 무덤",
            alias_candidates=["하페즈의 무덤"],
        ),
        "draft_text": "하페즈의 무덤은 どこに 있습니까?",
    },
    {
        "name": "Normal 케이스",
        "source": "How would you describe the material used for the Stele of the Vultures?",
        "target_lang": "fr",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Stele of the Vultures",
            qid="Q1088425",
            canonical_target="stèle des Vautours",
            alias_candidates=["stèle des Vautours"],
        ),
        "draft_text": "Comment décririez-vous les matériaux utilisés pour la stèle des Vautours?",
    },
    {
        "name": "Omission 케이스",
        "source": "In what country is the Stockholm Court House located?",
        "target_lang": "fr",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Stockholm Court House",
            qid="Q1112746",
            canonical_target="palais de justice de Stockholm",
            alias_candidates=["palais de justice de Stockholm"],
        ),
        "draft_text": "Dans quel pays se trouve le ?",
    },
    {
        "name": "Residue 케이스",
        "source": "How would you describe the material used for the Stele of the Vultures?",
        "target_lang": "fr",
        "memory": EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="Stele of the Vultures",
            qid="Q1088425",
            canonical_target="stèle des Vautours",
            alias_candidates=["stèle des Vautours"],
        ),
        "draft_text": "Comment décririez-vous les 가 matériaux utilisés pour la stèle des Vautours?",
    },
]

print("모델 로딩 시작")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=True,
)
print("모델 로딩 완료\n")

results = []

for tc in test_cases:
    print(f"{'='*50}")
    print(f"[{tc['name']}]")
    print(f"source     : {tc['source']}")
    print(f"draft_text : {tc['draft_text']}")
 
    draft = TranslationDraft(
        prompt_text="",
        draft_text=tc["draft_text"],
        raw_generation=tc["draft_text"],
        used_memory=tc["memory"],
    )
 
    alias_set = build_alias_set(tc["memory"])
    print(f"alias_set  : {alias_set}")


    decision = run_llm_judge(
        source=tc["source"],
        draft=draft,
        target_lang=tc["target_lang"],
        alias_set=alias_set,
        llm=llm,
    )
 
    print(f"should_run  : {decision.should_run}")
    print(f"error_types : {decision.error_types}")
    print(f"reasons     : {decision.reasons}")
    print()

    results.append({
        "name": tc["name"],
        "source": tc["source"],
        "draft_text": tc["draft_text"],
        "target_lang": tc["target_lang"],
        "alias_set": alias_set,
        "should_run": decision.should_run,
        "error_types": decision.error_types,
        "reasons": decision.reasons,
    })

output_path = os.path.join(os.path.dirname(__file__), "test_judge_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
 
print(f"결과 저장 완료: {output_path}")