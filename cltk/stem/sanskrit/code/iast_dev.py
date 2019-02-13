"""
This module holds a converting script for IAST >>> Devanāgarī.
"""

"""
    Copyright (c) 2017 Caio Borges.
    This file is part of L-om.
    L-om is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

'''Dictionaries for IAST >> DV'''
iast_dv_unicode = {'ṃ': '\u0902', 'ḥ': '\u0903',
                   'a': '\u0905', 'ā': '\u0906',
                   'i': '\u0907', 'ī': '\u0908',
                   'u': '\u0909', 'ū': '\u090A',
                   'ṛ': '\u090B', 'ṝ': 'ॠ',
                   'ḷ': '\u090C',
                   'e': '\u090F', 'ai': '\u0910',
                   'o': '\u0913', 'au': '\u0914',
                   'k': '\u0915', 'kh': '\u0916', 'g': '\u0917', 'gh': '\u0918', 'ṅ': '\u0919',
                   'c': '\u091A', 'ch': '\u091B', 'j': '\u091C', 'jh': '\u091D', 'ñ': '\u091E',
                   'ṭ': '\u091F', 'ṭh': '\u0920', 'ḍ': '\u0921', 'ḍh': '\u0922', 'ṇ': '\u0923',
                   't': '\u0924', 'th': '\u0925', 'd': '\u0926', 'dh': '\u0927', 'n': '\u0928',
                   'p': '\u092A', 'ph': '\u092B', 'b': '\u092C', 'bh': '\u092D', 'm': '\u092E',
                   'y': '\u092F', 'r': '\u0930', 'l': '\u0932', 'v': '\u0935',
                   'ś': '\u0936', 'ṣ': '\u0937', 's': '\u0938', 'h': '\u0939',
                   "'": '\u093D', 'oṃ': '\u0950', ' ': ' ', '|': '\u0964', '||': '\u0964'
                   }
iast_dv_vowel_diacritics = {'ā': '\u093E', 'a': '',
                            'i': '\u093F', 'ī': '\u0940',
                            'u': '\u0941', 'ū': '\u0942',
                            'ṛ': '\u0943', 'ṝ': '\u0944',
                            'ḷ': '\u0962',
                            'e': '\u0947', 'ai': '\u0948',
                            'o': '\u094b', 'au': '\u094C'}
iast_dv_vowels = ['ā', 'a', 'i', 'ī', 'u', 'ū', 'ṛ', 'ṝ', 'ḷ', 'e', 'ai', 'o', 'au']
iast_dv_consonants = ['k', 'g', 'ṅ',
                      'c', 'j', 'ñ',
                      'ṭ', 'ḍ', 'ṇ',
                      't', 'd', 'n',
                      'p', 'b', 'm',
                      'y', 'r', 'l', 'v',
                      'ś', 'ṣ', 's', 'h']
iast_dv_special_consonants = ['ṅ', 'ñ', 'ṇ', 'n', 'm',
                              'y', 'r', 'l', 'v',
                              'ś', 'ṣ', 's']
iast_dv_diacritics = ['ṃ', 'ḥ', '\'']
iast_dv_punctuation = ['|', '||']


def iastdv(input_text):

    """
    :param input_text: Text to be transliterated, written in IAST.
    :return: Text in Devanagari.
    """

    if input_text[-1] != ' ':
        input_text = input_text.center(len(input_text) + 2)
    output_text = []
    i = -1

    for letter in input_text:
            i += 1
            if letter not in iast_dv_unicode:
                output_text.append(letter)
            else:
                if letter in iast_dv_vowels:
                    if letter == 'a':
                        if input_text[i + 1] == 'i':
                            if input_text[i - 1] in iast_dv_consonants:
                                output_text.append(iast_dv_vowel_diacritics['ai'])
                            else:
                                output_text.append(iast_dv_unicode['ai'])
                        elif input_text[i + 1] == 'u':
                            if input_text[i - 1] in iast_dv_consonants:
                                output_text.append(iast_dv_vowel_diacritics['au'])
                            else:
                                output_text.append(iast_dv_unicode['au'])
                        else:
                            if input_text[i - 1] in iast_dv_consonants:
                                output_text.append('')
                            else:
                                output_text.append(iast_dv_unicode[letter])
                    elif letter == 'i' or letter == 'u':
                        if input_text[i - 1] == 'a':
                            continue
                        else:
                            if input_text[i - 1] in iast_dv_consonants:
                                output_text.append(iast_dv_vowel_diacritics[letter])
                            else:
                                output_text.append(iast_dv_unicode[letter])
                    else:
                        if input_text[i - 1] in iast_dv_consonants:
                            output_text.append(iast_dv_vowel_diacritics[letter])
                        else:
                            output_text.append(iast_dv_unicode[letter])
                elif letter in iast_dv_consonants:
                    if input_text[i + 1] == 'h' and letter not in iast_dv_special_consonants:
                        output_text.append(iast_dv_unicode['%sh' % letter])
                    elif letter == 'h' and input_text[i - 1] in iast_dv_consonants:
                        if input_text[i+1] not in iast_dv_vowels:
                                output_text.append('\u094D')
                    elif input_text[i + 1] not in iast_dv_vowels:
                        output_text.append(iast_dv_unicode[letter])
                        output_text.append('\u094D')
                    else:
                        output_text.append(iast_dv_unicode[letter])
                else:
                    output_text.append(iast_dv_unicode[letter])

    output_text = "".join(output_text)
    output_text = output_text.strip()
    return output_text

if __name__ == '__main__':
    print(iastdv("uktam"))