{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "14c65c1d-677f-4f2d-9b63-fe19654d11e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import BartTokenizer, BartForConditionalGeneration\n",
    "\n",
    "# Load the BART tokenizer and model\n",
    "tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')\n",
    "model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "caef2445-c0bd-4e1e-98f6-0a309ec7d910",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def summarize(text, max_length=150, min_length=30, num_beams=4):\n",
    "    \"\"\"\n",
    "    Summarizes the given text using the BART model.\n",
    "    \n",
    "    Parameters:\n",
    "    text (str): The text to be summarized.\n",
    "    max_length (int): The maximum length of the summary.\n",
    "    min_length (int): The minimum length of the summary.\n",
    "    num_beams (int): The number of beams for beam search. More beams result in better performance but are slower.\n",
    "    \n",
    "    Returns:\n",
    "    str: The generated summary.\n",
    "    \"\"\"\n",
    "    # Tokenize the input text\n",
    "    inputs = tokenizer.encode(\"summarize: \" + text, return_tensors='pt', max_length=512, truncation=True)\n",
    "    \n",
    "    # Generate the summary\n",
    "    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=num_beams, length_penalty=2.0, early_stopping=True)\n",
    "    \n",
    "    # Decode the summary\n",
    "    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)\n",
    "    return summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7b5f9938-8694-4ccd-8894-13eeec6f9e43",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Text:\n",
      " \n",
      "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term \"artificial intelligence\" is often used to describe machines (or computers) that mimic \"cognitive\" functions that humans associate with the human mind, such as \"learning\" and \"problem-solving\".\n",
      "As machines become increasingly capable, tasks considered to require \"intelligence\" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, intelligent routing in content delivery networks, and military simulations.\n",
      "\n",
      "\n",
      "Summary:\n",
      " Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems.\n"
     ]
    }
   ],
   "source": [
    "text = \"\"\"\n",
    "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term \"artificial intelligence\" is often used to describe machines (or computers) that mimic \"cognitive\" functions that humans associate with the human mind, such as \"learning\" and \"problem-solving\".\n",
    "As machines become increasingly capable, tasks considered to require \"intelligence\" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, intelligent routing in content delivery networks, and military simulations.\n",
    "\"\"\"\n",
    "\n",
    "summary = summarize(text)\n",
    "print(\"Original Text:\\n\", text)\n",
    "print(\"\\nSummary:\\n\", summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f583535f-bc5f-4a6c-aca0-6d349836d4f6",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Text:\n",
      " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed \"in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014.\" Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move toward greater justice. \"As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice,\" he said, according to an ICC news release. \"Indeed, today brings us closer to our shared goals of justice and peace.\" Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. \"As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly,\" she said. Rights group Human Rights Watch welcomed the development. \"Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court's treaty should speak out to welcome its membership,\" said Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members.\" In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it \"strongly\" disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC,\" the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. \"We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,\" it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as \"Palestine.\" While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would \"conduct its analysis in full independence and impartiality.\" The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.\n",
      "\n",
      "Reference Summary:\n",
      " Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .\n",
      "Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis .\n",
      "\n",
      "Generated Summary:\n",
      " The Palestinian Authority officially becomes the 123rd member of the International Criminal Court. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based.\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "# Load the CNN/DailyMail dataset\n",
    "dataset = load_dataset('cnn_dailymail', '3.0.0')\n",
    "\n",
    "# Get an example article and its reference summary\n",
    "example = dataset['test'][0]\n",
    "article = example['article']\n",
    "highlights = example['highlights']\n",
    "\n",
    "# Summarize the article\n",
    "summary = summarize(article)\n",
    "print(\"Original Text:\\n\", article)\n",
    "print(\"\\nReference Summary:\\n\", highlights)\n",
    "print(\"\\nGenerated Summary:\\n\", summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1aff6c7a-ee9c-4229-bdee-f61ee927f509",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROUGE Scores:\n",
      " {'rouge1': Score(precision=0.1875, recall=0.17647058823529413, fmeasure=0.1818181818181818), 'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rougeL': Score(precision=0.15625, recall=0.14705882352941177, fmeasure=0.15151515151515152)}\n"
     ]
    }
   ],
   "source": [
    "from rouge_score import rouge_scorer\n",
    "\n",
    "def compute_rouge_scores(reference, generated):\n",
    "    \"\"\"\n",
    "    Computes ROUGE scores between the reference and generated summaries.\n",
    "    \n",
    "    Parameters:\n",
    "    reference (str): The reference summary.\n",
    "    generated (str): The generated summary.\n",
    "    \n",
    "    Returns:\n",
    "    dict: A dictionary containing the ROUGE scores.\n",
    "    \"\"\"\n",
    "    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)\n",
    "    scores = scorer.score(reference, generated)\n",
    "    return scores\n",
    "\n",
    "# Compute ROUGE scores\n",
    "rouge_scores = compute_rouge_scores(highlights, summary)\n",
    "print(\"ROUGE Scores:\\n\", rouge_scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0f503b4a-148e-403d-8bdd-787102072f30",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Sample 1:\n",
      "Original Text:\n",
      " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed \"in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014.\" Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move toward greater justice. \"As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice,\" he said, according to an ICC news release. \"Indeed, today brings us closer to our shared goals of justice and peace.\" Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. \"As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly,\" she said. Rights group Human Rights Watch welcomed the development. \"Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court's treaty should speak out to welcome its membership,\" said Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members.\" In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it \"strongly\" disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC,\" the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. \"We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,\" it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as \"Palestine.\" While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would \"conduct its analysis in full independence and impartiality.\" The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.\n",
      "\n",
      "Reference Summary:\n",
      " Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .\n",
      "Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis .\n",
      "\n",
      "Generated Summary:\n",
      " The Palestinian Authority officially becomes the 123rd member of the International Criminal Court. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based.\n",
      "\n",
      "ROUGE Scores:\n",
      " {'rouge1': Score(precision=0.1875, recall=0.17647058823529413, fmeasure=0.1818181818181818), 'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rougeL': Score(precision=0.15625, recall=0.14705882352941177, fmeasure=0.15151515151515152)}\n",
      "\n",
      "Sample 2:\n",
      "Original Text:\n",
      " (CNN)Never mind cats having nine lives. A stray pooch in Washington State has used up at least three of her own after being hit by a car, apparently whacked on the head with a hammer in a misguided mercy killing and then buried in a field -- only to survive. That's according to Washington State University, where the dog -- a friendly white-and-black bully breed mix now named Theia -- has been receiving care at the Veterinary Teaching Hospital. Four days after her apparent death, the dog managed to stagger to a nearby farm, dirt-covered and emaciated, where she was found by a worker who took her to a vet for help. She was taken in by Moses Lake, Washington, resident Sara Mellado. \"Considering everything that she's been through, she's incredibly gentle and loving,\" Mellado said, according to WSU News. \"She's a true miracle dog and she deserves a good life.\" Theia is only one year old but the dog's brush with death did not leave her unscathed. She suffered a dislocated jaw, leg injuries and a caved-in sinus cavity -- and still requires surgery to help her breathe. The veterinary hospital's Good Samaritan Fund committee awarded some money to help pay for the dog's treatment, but Mellado has set up a fundraising page to help meet the remaining cost of the dog's care. She's also created a Facebook page to keep supporters updated. Donors have already surpassed the $10,000 target, inspired by Theia's tale of survival against the odds. On the fundraising page, Mellado writes, \"She is in desperate need of extensive medical procedures to fix her nasal damage and reset her jaw. I agreed to foster her until she finally found a loving home.\" She is dedicated to making sure Theia gets the medical attention she needs, Mellado adds, and wants to \"make sure she gets placed in a family where this will never happen to her again!\" Any additional funds raised will be \"paid forward\" to help other animals. Theia is not the only animal to apparently rise from the grave in recent weeks. A cat in Tampa, Florida, found seemingly dead after he was hit by a car in January, showed up alive in a neighbor's yard five days after he was buried by his owner. The cat was in bad shape, with maggots covering open wounds on his body and a ruined left eye, but remarkably survived with the help of treatment from the Humane Society.\n",
      "\n",
      "Reference Summary:\n",
      " Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer and buried in a field .\n",
      "\"She's a true miracle dog and she deserves a good life,\" says Sara Mellado, who is looking for a home for Theia .\n",
      "\n",
      "Generated Summary:\n",
      " Theia, a one-year-old bully breed mix, was hit by a car and buried in a field. She managed to stagger to a nearby farm, dirt-covered and emaciated. She suffered a dislocated jaw, leg injuries and a caved-in sinus cavity -- and still requires surgery to help her breathe.\n",
      "\n",
      "ROUGE Scores:\n",
      " {'rouge1': Score(precision=0.4117647058823529, recall=0.4883720930232558, fmeasure=0.44680851063829785), 'rouge2': Score(precision=0.24, recall=0.2857142857142857, fmeasure=0.2608695652173913), 'rougeL': Score(precision=0.4117647058823529, recall=0.4883720930232558, fmeasure=0.44680851063829785)}\n",
      "\n",
      "Sample 3:\n",
      "Original Text:\n",
      " (CNN)If you've been following the news lately, there are certain things you doubtless know about Mohammad Javad Zarif. He is, of course, the Iranian foreign minister. He has been U.S. Secretary of State John Kerry's opposite number in securing a breakthrough in nuclear discussions that could lead to an end to sanctions against Iran -- if the details can be worked out in the coming weeks. And he received a hero's welcome as he arrived in Iran on a sunny Friday morning. \"Long live Zarif,\" crowds chanted as his car rolled slowly down the packed street. You may well have read that he is \"polished\" and, unusually for one burdened with such weighty issues, \"jovial.\" An Internet search for \"Mohammad Javad Zarif\" and \"jovial\" yields thousands of results. He certainly has gone a long way to bring Iran in from the cold and allow it to rejoin the international community. But there are some facts about Zarif that are less well-known. Here are six: . In September 2013, Zarif tweeted \"Happy Rosh Hashanah,\" referring to the Jewish New Year. That prompted Christine Pelosi, the daughter of House Minority Leader Nancy Pelosi, to respond with a tweet of her own: \"Thanks. The New Year would be even sweeter if you would end Iran's Holocaust denial, sir.\" And, perhaps to her surprise, Pelosi got a response. \"Iran never denied it,\" Zarif tweeted back. \"The man who was perceived to be denying it is now gone. Happy New Year.\" The reference was likely to former Iranian President Mahmoud Ahmadinejad, who had left office the previous month. Zarif was nominated to be foreign minister by Ahmadinejad's successor, Hassan Rouhami. His foreign ministry notes, perhaps defensively, that \"due to the political and security conditions of the time, he decided to continue his education in the United States.\" That is another way of saying that he was outside the country during the demonstrations against the Shah of Iran, which began in 1977, and during the Iranian Revolution, which drove the shah from power in 1979. Zarif left the country in 1977, received his undergraduate degree from San Francisco State University in 1981, his master's in international relations from the University of Denver in 1984 and his doctorate from the University of Denver in 1988. Both of his children were born in the United States. The website of the Iranian Foreign Ministry, which Zarif runs, cannot even agree with itself on when he was born. The first sentence of his official biography, perhaps in a nod to the powers that be in Tehran, says Zarif was \"born to a religious traditional family in Tehran in 1959.\" Later on the same page, however, his date of birth is listed as January 8, 1960. And the Iranian Diplomacy website says he was born in in 1961 . So he is 54, 55 or maybe even 56. Whichever, he is still considerably younger than his opposite number, Kerry, who is 71. The feds investigated him over his alleged role in controlling the Alavi Foundation, a charitable organization. The U.S. Justice Department said the organization was secretly run on behalf of the Iranian government to launder money and get around U.S. sanctions. But last year, a settlement in the case, under which the foundation agreed to give a 36-story building in Manhattan along with other properties to the U.S. government, did not mention Zarif's name. Early in the Iranian Revolution, Zarif was among the students who took over the Iranian Consulate in San Francisco. The aim, says the website Iranian.com -- which cites Zarif's memoirs, titled \"Mr. Ambassador\" -- was to expel from the consulate people who were not sufficiently Islamic. Later, the website says, Zarif went to make a similar protest at the Iranian mission to the United Nations. In response, the Iranian ambassador to the United Nations offered him a job. In fact, he has now spent more time with Kerry than any other foreign minister in the world. And that amount of quality time will only increase as the two men, with help from other foreign ministers as well, try to meet a June 30 deadline for nailing down the details of the agreement they managed to outline this week in Switzerland.\n",
      "\n",
      "Reference Summary:\n",
      " Mohammad Javad Zarif has spent more time with John Kerry than any other foreign minister .\n",
      "He once participated in a takeover of the Iranian Consulate in San Francisco .\n",
      "The Iranian foreign minister tweets in English .\n",
      "\n",
      "Generated Summary:\n",
      " Mohammad Javad Zarif is the Iranian foreign minister. He has helped secure a breakthrough in nuclear discussions with the U.S. He received his undergraduate degree from San Francisco State University in 1981.\n",
      "\n",
      "ROUGE Scores:\n",
      " {'rouge1': Score(precision=0.48484848484848486, recall=0.45714285714285713, fmeasure=0.4705882352941177), 'rouge2': Score(precision=0.21875, recall=0.20588235294117646, fmeasure=0.21212121212121213), 'rougeL': Score(precision=0.3333333333333333, recall=0.3142857142857143, fmeasure=0.3235294117647059)}\n",
      "\n",
      "Sample 4:\n",
      "Original Text:\n",
      " (CNN)Five Americans who were monitored for three weeks at an Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, a Nebraska Medicine spokesman said in an email Wednesday. One of the five had a heart-related issue on Saturday and has been discharged but hasn't left the area, Taylor Wilson wrote. The others have already gone home. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus. They are clinicians for Partners in Health, a Boston-based aid group. They all had contact with a colleague who was diagnosed with the disease and is being treated at the National Institutes of Health in Bethesda, Maryland. As of Monday, that health care worker is in fair condition. The Centers for Disease Control and Prevention in Atlanta has said the last of 17 patients who were being monitored are expected to be released by Thursday. More than 10,000 people have died in a West African epidemic of Ebola that dates to December 2013, according to the World Health Organization. Almost all the deaths have been in Guinea, Liberia and Sierra Leone. Ebola is spread by direct contact with the bodily fluids of an infected person.\n",
      "\n",
      "Reference Summary:\n",
      " 17 Americans were exposed to the Ebola virus while in Sierra Leone in March .\n",
      "Another person was diagnosed with the disease and taken to hospital in Maryland .\n",
      "National Institutes of Health says the patient is in fair condition after weeks of treatment .\n",
      "\n",
      "Generated Summary:\n",
      " One of the five had a heart-related issue on Saturday and has been discharged. The others have already gone home. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus.\n",
      "\n",
      "ROUGE Scores:\n",
      " {'rouge1': Score(precision=0.40540540540540543, recall=0.35714285714285715, fmeasure=0.379746835443038), 'rouge2': Score(precision=0.16666666666666666, recall=0.14634146341463414, fmeasure=0.15584415584415584), 'rougeL': Score(precision=0.2702702702702703, recall=0.23809523809523808, fmeasure=0.25316455696202533)}\n",
      "\n",
      "Sample 5:\n",
      "Original Text:\n",
      " (CNN)A Duke student has admitted to hanging a noose made of rope from a tree near a student union, university officials said Thursday. The prestigious private school didn't identify the student, citing federal privacy laws. In a news release, it said the student was no longer on campus and will face student conduct review. The student was identified during an investigation by campus police and the office of student affairs and admitted to placing the noose on the tree early Wednesday, the university said. Officials are still trying to determine if other people were involved. Criminal investigations into the incident are ongoing as well. Students and faculty members marched Wednesday afternoon chanting \"We are not afraid. We stand together,\"  after pictures of the noose were passed around on social media. At a forum held on the steps of Duke Chapel, close to where the noose was discovered at 2 a.m., hundreds of people gathered. \"You came here for the reason that you want to say with me, 'This is no Duke we will accept. This is no Duke we want. This is not the Duke we're here to experience. And this is not the Duke we're here to create,' \" Duke President Richard Brodhead told the crowd. The incident is one of several recent racist events to affect college students. Last month a fraternity at the University of Oklahoma had its charter removed after a video surfaced showing members using the N-word and referring to lynching in a chant. Two students were expelled. In February, a noose was hung around the neck of a statue of a famous civil rights figure at the University of Mississippi. A statement issued by Duke said there was a previous report of hate speech directed at students on campus. In the news release, the vice president for student affairs called the noose incident a \"cowardly act.\" \"To whomever committed this hateful and stupid act, I just want to say that if your intent was to create fear, it will have the opposite effect,\" Larry Moneta said Wednesday. Duke University is a private college with about 15,000 students in Durham, North Carolina. CNN's Dave Alsup contributed to this report.\n",
      "\n",
      "Reference Summary:\n",
      " Student is no longer on Duke University campus and will face disciplinary review .\n",
      "School officials identified student during investigation and the person admitted to hanging the noose, Duke says .\n",
      "The noose, made of rope, was discovered on campus about 2 a.m.\n",
      "\n",
      "Generated Summary:\n",
      " A student has admitted to hanging a noose from a tree near a student union, university officials say. The student was identified during an investigation by campus police and the office of student affairs. Officials are still trying to determine if other people were involved.\n",
      "\n",
      "ROUGE Scores:\n",
      " {'rouge1': Score(precision=0.4222222222222222, recall=0.4523809523809524, fmeasure=0.4367816091954023), 'rouge2': Score(precision=0.09090909090909091, recall=0.0975609756097561, fmeasure=0.09411764705882353), 'rougeL': Score(precision=0.2, recall=0.21428571428571427, fmeasure=0.20689655172413796)}\n"
     ]
    }
   ],
   "source": [
    "def batch_summarize_and_evaluate(dataset, num_samples=10):\n",
    "    \"\"\"\n",
    "    Summarizes multiple articles and evaluates using ROUGE scores.\n",
    "    \n",
    "    Parameters:\n",
    "    dataset: The dataset containing articles and summaries.\n",
    "    num_samples (int): The number of samples to summarize and evaluate.\n",
    "    \n",
    "    Returns:\n",
    "    list: A list of dictionaries containing the original text, reference summary, generated summary, and ROUGE scores.\n",
    "    \"\"\"\n",
    "    results = []\n",
    "    for i in range(num_samples):\n",
    "        example = dataset['test'][i]\n",
    "        article = example['article']\n",
    "        highlights = example['highlights']\n",
    "        \n",
    "        summary = summarize(article)\n",
    "        rouge_scores = compute_rouge_scores(highlights, summary)\n",
    "        \n",
    "        results.append({\n",
    "            'article': article,\n",
    "            'reference_summary': highlights,\n",
    "            'generated_summary': summary,\n",
    "            'rouge_scores': rouge_scores\n",
    "        })\n",
    "    \n",
    "    return results\n",
    "\n",
    "# Summarize and evaluate multiple samples\n",
    "results = batch_summarize_and_evaluate(dataset, num_samples=5)\n",
    "\n",
    "# Print results\n",
    "for i, result in enumerate(results):\n",
    "    print(f\"\\nSample {i+1}:\")\n",
    "    print(\"Original Text:\\n\", result['article'])\n",
    "    print(\"\\nReference Summary:\\n\", result['reference_summary'])\n",
    "    print(\"\\nGenerated Summary:\\n\", result['generated_summary'])\n",
    "    print(\"\\nROUGE Scores:\\n\", result['rouge_scores'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a731f343-31a8-4884-a70f-c2298bf86295",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-07-18 03:54:32.787 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.11/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-07-18 03:54:32.787 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# Save this as app.py\n",
    "import streamlit as st\n",
    "\n",
    "# Define the summarization function\n",
    "def summarize(text, max_length=150, min_length=30, num_beams=4):\n",
    "    inputs = tokenizer.encode(\"summarize: \" + text, return_tensors='pt', max_length=512, truncation=True)\n",
    "    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=num_beams, length_penalty=2.0, early_stopping=True)\n",
    "    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)\n",
    "    return summary\n",
    "\n",
    "# Streamlit app\n",
    "st.title(\"Text Summarization with BART\")\n",
    "text = st.text_area(\"Enter text to summarize\", height=200)\n",
    "if st.button(\"Summarize\"):\n",
    "    summary = summarize(text)\n",
    "    st.write(\"**Summary:**\")\n",
    "    st.write(summary)\n",
    "\n",
    "# Run the Streamlit app\n",
    "# !streamlit run app.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "9655f82e-6601-461a-9dcb-fb7f9a6efb6d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
