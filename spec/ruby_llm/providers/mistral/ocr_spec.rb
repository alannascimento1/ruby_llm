# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Providers::Mistral::OCR do
  include_context 'with configured RubyLLM'

  let(:image_path) { File.expand_path('../../../fixtures/ruby.png', __dir__) }
  let(:ocr_endpoint) { 'https://api.mistral.ai/v1/ocr' }
  let(:model_id) { 'mistral-ocr-2505' }

  describe 'chat integration' do
    it 'sends OCR payloads to the dedicated endpoint and returns aggregated text' do
      stub_request(:post, ocr_endpoint)
        .with(
          headers: { 'Authorization' => 'Bearer test' },
          body: hash_including(
            model: model_id,
            document: hash_including(
              type: 'image_url',
              image_url: a_string_matching(%r{\Adata:image/png;base64,})
            )
          )
        )
        .to_return(
          status: 200,
          body: {
            results: [
              {
                text: 'Line One',
                blocks: [
                  { text: 'Line One' },
                  { text: 'Line Two' }
                ]
              }
            ],
            usage: {
              prompt_tokens: 321,
              completion_tokens: 42
            }
          }.to_json,
          headers: { 'Content-Type' => 'application/json' }
        )

      chat = RubyLLM.chat(model: model_id)
      response = chat.ask('extraia o texto', with: image_path)

      expect(response.content).to eq("Line One\n\nLine Two")
      expect(response.input_tokens).to eq(321)
      expect(response.output_tokens).to eq(42)
      expect(response.model_id).to eq(model_id)
      expect(a_request(:post, ocr_endpoint)).to have_been_made
    end

    it 'raises a friendly error when no attachment is provided' do
      chat = RubyLLM.chat(model: model_id)

      expect { chat.ask('extraia o texto') }
        .to raise_error(RubyLLM::Error, /require an attachment/i)
    end
  end
end
