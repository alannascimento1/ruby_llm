# frozen_string_literal: true

module RubyLLM
  module Providers
    # Mistral API integration.
    class Mistral < OpenAI
      include Mistral::Chat
      include Mistral::Models
      include Mistral::Embeddings
      include Mistral::OCR

      def api_base
        'https://api.mistral.ai/v1'
      end

      def headers
        {
          'Authorization' => "Bearer #{@config.mistral_api_key}"
        }
      end

      def complete(messages, tools:, temperature:, model:, params: {}, headers: {}, schema: nil, &) # rubocop:disable Metrics/ParameterLists
        if ocr_model?(model.id)
          raise RubyLLM::Error.new(nil, 'Mistral OCR models do not support tool calls.') if tools&.any?
          raise RubyLLM::Error.new(nil, 'Mistral OCR models do not support response schemas.') if schema
          raise RubyLLM::Error.new(nil, 'Mistral OCR models do not support streaming responses.') if block_given?

          return complete_ocr(
            messages,
            model:,
            params:,
            headers:,
            streaming: false
          )
        end

        super
      end

      class << self
        def capabilities
          Mistral::Capabilities
        end

        def configuration_requirements
          %i[mistral_api_key]
        end
      end
    end
  end
end
