# MSiA - Text Analytics
# Homework 1
# Luis Steven Lin

# Files:
- readme.txt
- regex.txt
- fsa.txt
- src/hw1.py
- test_reg_ex.txt

For class bios, absolute times (14:00 PST) do not make sense to extract. Instead focusing on absolute dates (10/12/2014) and relative times make more sense. Regular expressions for absolute dates are implemented in detail, however, it is not very likely that someone will use absolute dates in a formal format. Thus, for class bios regular expressions to capture some time point that is more ambiguous (December 2015 or “last fall”) were also considered as these are more likely to appear and also might have a useful application of text analytics involving a time component in the class bios. In addition, based on the following questions asked in the assignment, an ambiguous relative time is more likely to appear in class bios (e.g. Fall 2015, this quarter, last summer, in 2010, 2014-2015) and were matched. 

- where you’re from
- what brought you to your current graduate program (e.g., MSiA)?
- what did you do before this program?
- when did you become interested in text analytics?
- what are your plans for the future?

A test set file is included to show that the regular expressions are working properly since most patterns for matching absolute dates are not found in the class bios.

#### Regular Expression explanations (regex found in regex.txt)
The following patterns match all cases in the test file (test_reg_ex).txt, which contains various date formats found in https://en.wikipedia.org/wiki/Calendar_date.
Note that the actual code was improved using substrings to make the regular expressions more concise and reusable. The file regex.txt contains the raw, complete regular expressions to make it easier to test across different platforms. The logic behind the regular expressions is outlined below. 

# Pattern 1
# Example yyyy/dd/mm
yyyy/dd/mm
yyyy/dd/m
yyyy/d/mm
yyyy/d/m
yyyy/mm/dd
yyyy/mm/d
yyyy/m/dd
yyyy/d/d
Same pattern as above but with . or -

# Pattern 2
# Example mm/dd/yyyy
mm/dd/yyyy
mm/dd/yy
dd/mm/yyyy
dd/mm/yy
d/m/yyyy
d/m/yy
m/d/yyyy
m/d/yy
Same pattern as above but with . or -

# Pattern 3
# Example 2015Jan02

Note: one could add the following to take into account day of week, but the problem is asking to identify if a date exists, so extracting the day of week is not needed.
,? ?(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)?

# Pattern 4
# January 30th, 2015

# Pattern 5
# Example: 31st October, 2015

# Pattern 6
# Example: December 2015, fall 2005, in 2010

# Pattern 7
# Example: this fall, next spring, last summer, the past summer, 2014-2015


###### Code for regular expressions

### Substrings:
months_list = ["[Jj]an(?:uary)?","[Ff]eb(?:ruary)?", "[Mm]ar(?:ch)?",
          "[Aa]pr(?:il)?", "[Mm]ay","[Jj]un(?:e)?", "[Jj]ul(?:y)?",
          "[Aa]ug(?:ust)?","[Ss]ep(?:tember)?", "[Oo]ct(?:ober)?",
          "[Nn]ov(?:ember)?","[Dd]ec(?:ember)?" ]
          
seasons_list = ["[Ss]ummer","[Ff]all","[Ww]inter","[Ss]pring","[Qq]uarter]"]

adjectives_list = ["next", "past", "last", "this"]

months = "|".join(months_list)
seasons =  "|".join(seasons_list)
adjectives = "|".join(adjectives_list)

### Regular Expressions:

### Example yyyy/dd/mm
# Match a year followed by day and months (1 or 2 digit format) in any order with - or . or / separators

pattern1 = re.compile(r"\b[0-9]{4}[-/.][0-9]?[0-9][-/.][0-9]?[0-9]")

### Example mm/dd/yyyy
# Match a day and months in any order (1 or 2 digit format), followed by year 2 or 4 digit with - or . or / separators

pattern2 = re.compile(r"\b[0-3]?[0-9][/.-][0-3]?[0-9][/.-](?:[0-9]{2})?[0-9]{2}")

### Example 2015Jan02
# {{ }} to escape replacement
# Match year, followed by month name (with first letter uppercase or lowercase) followed by day (1 or 2 digit format)

pattern3 = re.compile(r"\b[0-9]{{4}}(?:[-/. ]?|. )(?:{months})[-/. ]?[0-9]?[0-9]".format(months=months))
 
### January 30th, 2015
# {{ }} to escape replacement
# Match month name (with first letter uppercase or lowercase) followed by day and quantifier, followed by comma and year (4 digit)

pattern4 = re.compile(r"\b(?:{months})[-/. ]?[ ]?[0-9]?[0-9] ?(?:st|nd|rd|th)?,? ?[0-9]{{4}}".format(months=months))

### Example: 31st October, 2015
# {{ }} to escape replacement
# Match day followed by day and quantifier, followed by month name (with first letter uppercase or lowercase)m followed by comma and year (4 digit)

pattern5 = re.compile(r"\b[0-9]?[0-9][ ]?(?:(?:th|st|nd|rd) (?:of)?)?"
                       "(?:[-/. ]?|. )(?:{months})"
                       "(?:[-/. ]| AD | BC )?(?:[0-9]{{2}})?[0-9]{{2}}".format(months=months))

### Example: December 2015, fall 2005, in 2010
# {{ }} to escape replacement
# Match month name (with first letter uppercase or lowercase) or season (with first letter uppercase or lowercase) or “in” followed by 4-digit year

pattern6 = re.compile(r"\b(?:{months}|"
                       "{seasons}|[Ii]n )[-/. ]?[0-9]{{4}}".format(months=months,seasons=seasons))
                      
### Example: this fall, next spring, last summer, the past summer, 2014-2015
# {{ }} to escape replacement
# Match adjective followed by season, or match a 4-digit year followed by “-“ followed by 4-digit year

pattern7 = re.compile(r"\b(?:{adjectives}) (?:{seasons})|\b[0-9]{{4}}-[0-9]{{4}}".
                        format(seasons=seasons, adjectives = adjectives))
