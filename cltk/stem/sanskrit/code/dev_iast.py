#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module holds a converting script for Devanāgarī >>> IAST.
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

'''Dictionaries for DV >> IAST'''
dv_iast_unicode = {'ट': 'ṭ', 'ण': 'ṇ', 'ऊ': 'ū', 'ऋ': 'ṛ',
                   'श': 'ś', 'द': 'd', 'उ': 'u', 'ग': 'g',
                   'ज': 'j', 'थ': 'th', 'अ': 'a', 'आ': 'ā',
                   'ञ': 'ñ', 'ब': 'b', 'ङ': 'ṅ', 'म': 'm',
                   'न': 'n', 'ः': 'ḥ', 'त': 't', 'ष': 'ṣ',
                   'छ': 'ch', 'ढ': 'ḍh', 'ठ': 'ṭh', 'क': 'k',
                   'इ': 'i', 'ख': 'kh', 'च': 'c', 'ध': 'dh',
                   'ऐ': 'ai', 'ह': 'h', 'घ': 'gh', 'व': 'v',
                   'ओ': 'o', ' ': ' ', 'ऽ': "'", 'ॐ': 'oṃ',
                   'ड': 'ḍ', 'ई': 'ī', 'ए': 'e', 'झ': 'jh',
                   'ऌ': 'ḷ', 'य': 'y', '।': '|', 'फ': 'ph',
                   'औ': 'au', 'ल': 'l', 'स': 's', 'भ': 'bh',
                   'प': 'p', 'ं': 'ṃ', 'र': 'r'}
dv_iast_vowel_diacritics = {'ी': 'ī', 'े': 'e', 'ॄ': 'ṝ', 'ृ': 'ṛ',
                            'ै': 'ai', 'ॢ': 'ḷ', 'ो': 'o', 'ौ': 'au',
                            'ा': 'ā', 'ि': 'i', 'ु': 'u', 'ू': 'ū'}
dv_iast_vowels = ['ी', 'े', 'ॄ', 'ृ', 'ै', 'ॢ', 'ो', 'ौ', 'ा', 'ि', 'ु', 'ू']
dv_iast_consonants = ['क', 'ख', 'ग', 'घ', 'ङ',
                      'च', 'छ', 'ज', 'झ', 'ञ',
                      'ट', 'ठ', 'ड', 'ढ', 'ण',
                      'त', 'थ', 'द', 'ध', 'न',
                      'प', 'फ', 'ब', 'भ', 'म',
                      'य', 'र', 'ल', 'व',
                      'श', 'ष', 'स', 'ह']
dv_iast_diacritics = ['ं', 'ः', 'ऽ']
dv_iast_punctuation = ['।', '।।']


def dviast(input_text):

    """
    :param input_text: Text to be transliterated, written in Devanagari
    :return: Text in IAST
    """

    if input_text[-1] != ' ':
        input_text = input_text.center(len(input_text) + 2)
    output_text = []
    i = -1

    for letter in input_text:
        i += 1
        if letter not in dv_iast_unicode.keys() and letter not in dv_iast_vowels:
            output_text.append(letter)
            continue

        if i == len(input_text)-2:
            break

        if letter not in dv_iast_unicode and letter in dv_iast_vowels:
            output_text.append(dv_iast_vowel_diacritics[letter])
        elif letter in dv_iast_consonants:
            if input_text[i + 1] in dv_iast_vowel_diacritics:
                output_text.append(dv_iast_unicode[letter])
            elif input_text[i + 1] == ' ' or input_text[i + 1] in dv_iast_consonants:
                output_text.append(dv_iast_unicode[letter])
                output_text.append('a')
            elif input_text[i + 1] in dv_iast_diacritics or input_text[i + 1] in dv_iast_punctuation:
                output_text.append(dv_iast_unicode[letter])
                output_text.append('a')
            else:
                if input_text[i + 1] == '्':
                    output_text.append(dv_iast_unicode[letter])
                else:
                    print('epa')
        else:
            output_text.append(dv_iast_unicode[letter])

        for a in output_text:
            if a == '्':
                output_text.pop(output_text.index(a))

    output_text = ''.join(output_text)
    output_text = output_text.strip()
    return output_text

if __name__ == '__main__':
    print(dviast("साधूक्तम्"))