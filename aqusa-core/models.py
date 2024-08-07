
# Classes: Stories, Defect, Project  

class Stories(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  title = db.Column(db.Text)
  external_id = db.Column(db.Integer)
  role = db.Column(db.Text)
  means = db.Column(db.Text)
  ends = db.Column(db.Text)
  project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
  defects = db.relationship('Defects', backref='story', lazy='dynamic', cascade='save-update, merge, delete')
  created_at = db.Column(db.DateTime, default=datetime.now)
  updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

  def __repr__(self):
    return '<story: %r, title=%s>' % (self.id, self.title)

  def serialize(self):
    class_dict = self.__dict__
    del class_dict['_sa_instance_state']
    return class_dict

  def create(title, external_id, project_id, analyze=False):
    story = Stories(title=title, external_id=external_id, project_id=project_id)
    db.session.add(story)
    db.session.commit()
    db.session.merge(story)
    story.chunk()
    if analyze: story.analyze()
    return story

  def save(self):
    db.session.add(self)
    db.session.commit()
    db.session.merge(self)
    return self

  def delete(self):
    db.session.delete(self)
    db.session.commit()

  def chunk(self):
    StoryChunker.chunk_story(self)
    return self

  def re_chunk(self):
    self.role = None
    self.means = None
    self.ends = None
    StoryChunker.chunk_story(self)
    return self

  def analyze(self):
    WellFormedAnalyzer.well_formed(self)
    Analyzer.atomic(self)
    Analyzer.unique(self)
    MinimalAnalyzer.minimal(self)
    Analyzer.uniform(self)
    return self

  def re_analyze(self):
    self.analyze()
    return self


class Projects(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(120), index=True, nullable=False)
  external_id = db.Column(db.Integer)
  format = db.Column(db.Text, nullable=True, default="As a,I'm able to,So that")
  create_comments = db.Column(db.Boolean)
  stories = db.relationship('Stories', backref='project', lazy='dynamic', cascade='save-update, merge, delete')
  defects = db.relationship('Defects', backref='project', lazy='dynamic')
  created_at = db.Column(db.DateTime, default=datetime.now)
  updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

  def __repr__(self):
    return '<Project: %r, name=%s>' % (self.id, self.name)

  def serialize(self):
    class_dict = self.__dict__
    del class_dict['_sa_instance_state']
    return class_dict

  def create(name):
    project = Project(name=name)
    db.session.add(project)
    db.session.commit()
    db.session.merge(project)
    return project

  def delete(self):
    db.session.delete(self)
    db.session.commit() 

  def save(self):
    db.session.add(self)
    db.session.commit()
    db.session.merge(self)
    return self

  def process_csv(self, path):
    stories = pandas.read_csv(path, header=-1)
    for story in stories[0]: 
      Stories.create(text=story, project_id=self.id)
    self.analyze()
    return None

  def get_common_format(self):
    most_common_format = []
    for chunk in ['role', 'means', 'ends']:
      chunks = [Analyzer.extract_indicator_phrases(getattr(story,chunk), chunk) for story in self.stories]
      chunks = list(filter(None, chunks))
      try:
        most_common_format += [Counter(chunks).most_common(1)[0][0].strip()]
      except:
        print('')
    self.format = ', '.join(most_common_format)
    if self.format == "": self.format = "As a, I want to, So that"
    self.save() 
    return "New format is: " + self.format

  def analyze(self):
    for story in self.stories.all():
      story.re_chunk()
    self.get_common_format()
    for story in self.stories.all():
      story.analyze()
    return self

class Defects(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  highlight = db.Column(db.Text, nullable=False)
  kind = db.Column(db.String(120), index=True,  nullable=False)
  subkind = db.Column(db.String(120), nullable=False)
  severity = db.Column(db.String(120), nullable=False)
  false_positive = db.Column(db.Boolean, default=False, nullable=False)
  story_id = db.Column(db.Integer, db.ForeignKey('stories.id'), nullable=False)
  project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
  comments = db.relationship('Comments', backref='defect', lazy='dynamic')
  created_at = db.Column(db.DateTime, default=datetime.now)
  updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

  def __repr__(self):
    return '<Defect: %s, highlight=%s, kind=%s>' % (self.id, self.highlight, self.kind)

  def create(highlight, kind, subkind, severity, story):
    defect = Defects(highlight=highlight, kind=kind, subkind=subkind, severity=severity, story_id=story.id, project_id=story.project.id)
    db.session.add(defect)
    db.session.commit()
    db.session.merge(defect)
    return defect

  def delete(self):
    db.session.delete(self)
    db.session.commit()

  def save(self):
    db.session.add(self)
    db.session.commit()
    db.session.merge(self)
    return self

  def create_unless_duplicate(highlight, kind, subkind, severity, story):
    project = story.project
    defect = Defects(highlight=highlight, kind=kind, subkind=subkind, severity=severity, story_id=story.id, project_id=project.id)
    duplicates = Defects.query.filter_by(highlight=highlight, kind=kind, subkind=subkind,
      severity=severity, story_id=story.id, project_id=project.id, false_positive=False).all()
    if duplicates:
      return 'duplicate'
    else:
      db.session.add(defect)
      db.session.merge(defect)
      db.session.commit()
      if project.create_comments == True:
        Defects.send_comment(os.environ['FRONTEND_URL'], str(defect.id))
      return defect

  def send_comment(url, defect_id):
    r = requests.get("%s/defects/%s/create_comments" % (url, defect_id))

  def correct_minor_issue(self):
    story = self.story
    CorrectDefect.correct_minor_issue(self)
    return story

class Comments(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  external_id = db.Column(db.String)
  text = db.Column(db.Text)
  defect_id = db.Column(db.Integer, db.ForeignKey('defects.id'), nullable=False)
  # user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  created_at = db.Column(db.DateTime, default=datetime.now)
  updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

  def delete(self):
    db.session.delete(self)
    db.session.commit()




class Analyzer:
  def atomic(story):
    for chunk in ['"role"', '"means"', '"ends"']:
      Analyzer.generate_defects('atomic', story, chunk=chunk)
    return story

  def unique(story):
    Analyzer.generate_defects('unique', story)
    return story

  def uniform(story):
    Analyzer.generate_defects('uniform', story)
    return story

  def generate_defects(kind, story, **kwargs):
    for kwarg in kwargs:
      exec(kwarg+'='+ str(kwargs[kwarg]))
    for defect_type in ERROR_KINDS[kind]:
      if eval(defect_type['rule']):
        Defects.create_unless_duplicate(eval(defect_type['highlight']), kind, defect_type['subkind'], defect_type['severity'], story)

  def inject_text(text, severity='medium'):
    return "<span class='highlight-text severity-" + severity + "'>%s</span>" % text

  def atomic_rule(chunk, kind):
    sentences_invalid = []
    if chunk: 
      for x in CONJUNCTIONS:
        if x in chunk.lower():
          if kind == 'means':
            for means in re.split(x, chunk, flags=re.IGNORECASE):
              if means:
                sentences_invalid.append(Analyzer.well_formed_content_rule(means, 'means', ['MEANS']))
          if kind == 'role':
            kontinue = True
            if x in ['&', '+']: kontinue = Analyzer.symbol_in_role_exception(chunk, x)
            if kontinue:
              for role in re.split(x, chunk, flags=re.IGNORECASE):
                if role:
                  sentences_invalid.append(Analyzer.well_formed_content_rule(role, "role", ["NP"]))
    return sentences_invalid.count(False) > 1

  def symbol_in_role_exception(chunk, conjunction):
    surrounding_words = Analyzer.get_surrounding_words(chunk, conjunction)
    exception = [False, False, False]
    exception[0] = Analyzer.space_before_or_after_conjunction(chunk, conjunction)
    exception[1] = Analyzer.surrounding_words_bigger_than(3, surrounding_words)
    exception[2] = Analyzer.surrounding_words_valid(surrounding_words)
    return exception.count(True) >= 2

  def space_before_or_after_conjunction(chunk, conjunction):
    idx = chunk.lower().index(conjunction.lower())
    space = chunk[idx-1].isspace() or chunk[idx+len(conjunction)].isspace()
    return space

  def get_surrounding_words(chunk, conjunction):
    parts = chunk.split(conjunction)
    words = []
    for index, part in enumerate(parts):
      if index % 2 == 0:
        words += [part.split()[-1]]
      else:
        words += [part.split()[0]]
    return words

  def surrounding_words_bigger_than(number, word_array):
    result = False
    for word in word_array:
      if len(word) > number: result = True
    return result

  def surrounding_words_valid(word_array):
    result = False
    for word in word_array:
      if not wordnet.synsets(word): result = True
    return result
    

  def identical_rule(story):
    identical_stories = Stories.query.filter((Stories.title==story.title) & (Stories.project_id == int(story.project_id))).all()
    identical_stories.remove(story)
    return (True if identical_stories else False)

  def highlight_text(story, word_array, severity):
    indices = []
    for word in word_array:
      if word in story.title.lower(): indices += [ [story.title.lower().index(word), word] ]
    return Analyzer.highlight_text_with_indices(story.title, indices, severity)

  def highlight_text_with_indices(text, indices, severity):
    indices.sort(reverse=True)
    for index, word in indices:
      text = text[:index] + "<span class='highlight-text severity-" + severity + "'>" + word + "</span>" + text[index+len(word):]
    return text

  # result indicates whether the story_part contains a well_formed error
  def well_formed_content_rule(story_part, kind, tags):
    result = Analyzer.content_chunk(story_part, kind)
    well_formed = True
    for tag in tags:
      for x in result.subtrees():
        if tag.upper() in x.label(): well_formed = False
    return well_formed

  def uniform_rule(story):
    project_format = story.project.format.split(',')
    chunks = []
    for chunk in ['role', 'means', 'ends']:
      chunks += [Analyzer.extract_indicator_phrases(getattr(story,chunk), chunk)]
    chunks = list(filter(None, chunks))
    chunks = [c.strip() for c in chunks]
    result = False
    if len(chunks) == 1: result = True
    for x in range(0,len(chunks)):
      if nltk.metrics.distance.edit_distance(chunks[x].lower(), project_format[x].lower()) > 3:
        result = True 
    return result

  def well_formed_content_highlight(story_part, kind):
    return str(Analyzer.content_chunk(story_part, kind))

  def content_chunk(chunk, kind):
    sentence = AQUSATagger.parse(chunk)[0]
    sentence = Analyzer.strip_indicators_pos(chunk, sentence, kind)
    sentence = Analyzer.replace_tag_of_special_words(sentence)
    cp = nltk.RegexpParser(CHUNK_GRAMMAR)
    result = cp.parse(sentence)
    return result

  def replace_tag_of_special_words(sentence):
    index = 0
    for word in sentence:
      if word[0] in SPECIAL_WORDS:  
        lst = list(sentence[index])
        lst[1] = SPECIAL_WORDS[word[0]]
        sentence[index] = tuple(lst)
      index+=1
    return sentence

  def extract_indicator_phrases(text, indicator_type):
    if text:
      indicator_phrase = []
      for indicator in eval(indicator_type.upper() + '_INDICATORS'):
        if re.compile('(%s)' % indicator.lower().replace('[, ]', '')).search(text.lower()): indicator_phrase += [indicator.replace('^', '').replace('[, ]', '')]
      return max(indicator_phrase, key=len) if indicator_phrase else None
    else:
      return text

  def strip_indicators_pos(text, pos_text, indicator_type):
    for indicator in eval(indicator_type.upper() + '_INDICATORS'):
      if indicator.lower().strip() in text.lower():
        indicator_words = nltk.word_tokenize(indicator)
        pos_text = [x for x in pos_text if x[0] not in indicator_words]
    return pos_text


class WellFormedAnalyzer:
  def well_formed(story):
    WellFormedAnalyzer.means(story)
    WellFormedAnalyzer.role(story)
    # WellFormedAnalyzer.means_comma(story)
    # WellFormedAnalyzer.ends_comma(story)
    # return story

  def means(story):
    if not story.means:
      Defects.create_unless_duplicate('Add what you want to achieve', 'well_formed', 'no_means', 'high', story )
    return story

  def role(story):
    if not story.role:
      Defects.create_unless_duplicate('Add for who this story is', 'well_formed', 'no_role', 'high', story )
    return story

  # def ends(story):
  #   if not story.ends:
  #     Defects.create_unless_duplicate('Optionally add what the purpose is off this story', 'well_formed', 'no_ends', 'medium', story )
  #   return story

  def means_comma(story):
    if story.role is not None and story.means is not None:
      if story.role.count(',') == 0:
        highlight = story.role + Analyzer.inject_text(',') + ' ' + story.means
        Defects.create_unless_duplicate(highlight, 'well_formed', 'no_means_comma', 'minor', story )
    return story

  def ends_comma(story):
    if story.means is not None and story.ends is not None:
      if story.means.count(',') == 0:
        highlight = story.means + Analyzer.inject_text(',') + ' ' + story.ends
        Defects.create_unless_duplicate(highlight, 'well_formed', 'no_ends_comma', 'minor', story )
    return story

class MinimalAnalyzer:
  def minimal(story):
    MinimalAnalyzer.punctuation(story)
    MinimalAnalyzer.brackets(story)
    MinimalAnalyzer.indicator_repetition(story)
    return story

  def punctuation(story):
    if any(re.compile('(\%s .)' % x).search(story.title.lower()) for x in PUNCTUATION):
      highlight = MinimalAnalyzer.punctuation_highlight(story, 'high')
      Defects.create_unless_duplicate(highlight, 'minimal', 'punctuation', 'high', story )
    return story

  def punctuation_highlight(story, severity):
    highlighted_text = story.title
    indices = []
    for word in PUNCTUATION:
      if re.search('(\%s .)' % word, story.title.lower()): indices += [ [story.title.index(word), word] ]
    first_punct = min(indices)
    highlighted_text = highlighted_text[:first_punct[0]] + "<span class='highlight-text severity-" + severity + "'>" + highlighted_text[first_punct[0]:] + "</span>"
    return highlighted_text

  def brackets(story):
    if any(re.compile('(\%s' % x[0] + '.*\%s(\W|\Z))' % x[1]).search(story.title.lower()) for x in BRACKETS):
      highlight = MinimalAnalyzer.brackets_highlight(story, 'high')
      Defects.create_unless_duplicate(highlight, 'minimal', 'brackets', 'high', story )
    return story.defects.all()

  def brackets_highlight(story, severity):
    highlighted_text = story.title
    matches = []
    for x in BRACKETS:
      split_string = '[^\%s' % x[1] + ']+\%s' % x[1]
      strings = re.findall(split_string, story.title)
      match_string = '(\%s' % x[0] + '.*\%s(\W|\Z))' % x[1]
      string_length = 0
      for string in strings:
        result = re.compile(match_string).search(string.lower())
        if result:
          span = tuple(map(operator.add, result.span(), (string_length, string_length)))
          matches += [ [span, result.group()] ]
        string_length += len(string)
    matches.sort(reverse=True)
    for index, word in matches:
      highlighted_text = highlighted_text[:index[0]] + "<span class='highlight-text severity-" +  severity + "'>" + word + "</span>" + highlighted_text[index[1]:]
    return highlighted_text

  def indicator_repetition(story):
    indicators = StoryChunker.detect_all_indicators(story)
    for indicator in indicators: indicators[indicator] = StoryChunker.remove_overlapping_tuples(indicators[indicator])
    indicators = MinimalAnalyzer.remove_indicator_repetition_exceptions(indicators, story)
    for indicator in indicators:
      if len(indicators[indicator]) >= 2:
        highlight = MinimalAnalyzer.indicator_repetition_highlight(story.title, indicators[indicator], 'high')
        Defects.create_unless_duplicate(highlight, 'minimal', 'indicator_repetition', 'high', story)
    return story
  
  def indicator_repetition_highlight(text, ranges, severity):
    indices = []
    for rang in ranges:
      indices += [[rang[0], text[ rang[0]:rang[1] ] ]]
    return Analyzer.highlight_text_with_indices(text, indices, severity)
  
  def remove_indicator_repetition_exceptions(indicators, story):
    # exception #1: means indicator after ends indicator
    for means_indicator in indicators['means']:
      for ends_indicator in indicators['ends']:
        if ends_indicator[1] >= means_indicator[0] and ends_indicator[1] <= means_indicator[1]:
          indicators['means'].remove(means_indicator)
    return indicators



class StoryChunker:
  def chunk_story(story):
    StoryChunker.chunk_on_indicators(story)
    if story.means is None:
      potential_means = story.title
      if story.role is not None:
        potential_means = potential_means.replace(story.role, "", 1).strip()
      if story.ends is not None:
        potential_means = potential_means.replace(story.ends, "", 1).strip()
      StoryChunker.means_tags_present(story, potential_means)
    return story.role, story.means, story.ends

  def chunk_on_indicators(story):
    indicators = StoryChunker.detect_indicators(story)
    if indicators['means'] is not None and indicators['ends'] is not None:
      indicators = StoryChunker.correct_erroneous_indicators(story, indicators)
    if indicators['role'] is not None and indicators['means'] is not None:
      story.role = story.title[indicators['role']:indicators['means']].strip()
      story.means = story.title[indicators['means']:indicators['ends']].strip()
    elif indicators['role'] is not None and indicators['means'] is None:
      role = StoryChunker.detect_indicator_phrase(story.title, 'role')
      text = StoryChunker.remove_special_characters(story.title)
      new_text = text.replace(role[1], '')

      sentence = Analyzer.content_chunk(new_text, 'role')
      NPs_after_role = StoryChunker.keep_if_NP(sentence)
      if NPs_after_role:
        story.role = story.title[indicators['role']:(len(role[1]) + 1 + len(NPs_after_role))].strip()
    if indicators['ends']: story.ends = story.title[indicators['ends']:None].strip()
    story.save()
    return story

  def detect_indicators(story):
    indicators = {'role': None, "means": None, 'ends': None}
    for indicator in indicators:
      indicator_phrase = StoryChunker.detect_indicator_phrase(story.title, indicator)
      if indicator_phrase[0]:
        indicators[indicator.lower()] = story.title.lower().index(indicator_phrase[1].lower())
    return indicators

  def detect_all_indicators(story):
    indicators = {'role': [], "means": [], 'ends': []}
    for indicator in indicators:
      for indicator_phrase in eval(indicator.upper() + '_INDICATORS'):  
        if story.title:
          for indicator_match in re.compile('(%s)' % indicator_phrase.lower()).finditer(story.title.lower()):
            indicators[indicator] += [indicator_match.span()]
    return indicators

  # get longest from overlapping indicator hits
  def remove_overlapping_tuples(tuple_list):
    for hit in tuple_list:
      duplicate_tuple_list = list(tuple_list)
      duplicate_tuple_list.remove(hit)
      for duplicate in duplicate_tuple_list:
        if hit[0] <= duplicate[0] and hit[1] >= duplicate[1]:
          tuple_list.remove(duplicate)
        elif hit[0] >= duplicate[0] and hit[1] <= duplicate[1]:
          tuple_list.remove(hit)
    return tuple_list

  def remove_special_characters(text):
    return ''.join( e if (e.isalnum() or e.isspace() or e == "\'" or e == "\"" ) else ' ' for e in text).strip()

  def detect_indicator_phrase(text, indicator_type):
    result = False
    detected_indicators = ['']
    no_special_char_text = StoryChunker.remove_special_characters(text)
    for indicator_phrase in eval(indicator_type.upper() + '_INDICATORS'):
      if text:
        if re.compile('(%s)' % indicator_phrase.lower()).search(no_special_char_text.lower()):
          stripped_indicator = indicator_phrase.replace('^', '').replace('[, ]', '').strip()
          if stripped_indicator.lower() in text.lower():
            result = True
            detected_indicators.append(stripped_indicator)
    return (result, max(detected_indicators, key=len))


  def keep_if_NP(parsed_tree):
    return_string = []
    for leaf in parsed_tree:
      if type(leaf) is not tuple:
        if leaf[0][0] == 'I':
          break
        elif leaf.label() == 'NP': 
          return_string.append(leaf[0][0])
        else:
          break
      elif leaf == (',', ','): return_string.append(',')
    return ' '.join(return_string)

  def means_tags_present(story, string):
    if not Analyzer.well_formed_content_rule(string, 'means', ['MEANS']):
      story.means = string
      story.save
    return story

  def correct_erroneous_indicators(story, indicators):
    # means is larger than ends
    if indicators['means'] > indicators['ends']:
      new_means = StoryChunker.detect_indicator_phrase(story.title[:indicators['ends']], 'means')
      #replication of #427 - refactor?
      if new_means[0]:
        indicators['means'] = story.title.lower().index(new_means[1].lower())
      else:
        indicators['means'] = None
    return indicators


class CorrectDefect:
  def correct_minor_issue(defect):
    story = defect.story
    eval('CorrectDefect.correct_%s(defect)' % defect.subkind)
    return story

  def correct_no_means_comma(defect):
    story = defect.story
    story.title = story.role + ', ' + story.means 
    if story.ends: story.title = story.title + ' ' + story.ends
    story.save()
    return story

  def correct_no_ends_comma(defect):
    story = defect.story
    story.title = story.means + ', ' + story.ends
    if story.role: story.title = story.role + ' ' + story.title
    story.save()
    return story